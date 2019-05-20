# Inverts from conserved variables to primitives.
# Based on Mignone & McKinney 2007, Noble et al 2006

import numpy as np

import pyopencl.array as cl_array
import pyopencl.clmath as clm
import loopy as lp


from pyHARM.loopy_tools import *
use_2018_2()

from pyHARM.defs import Loci

from datetime import datetime

knl_utop_iter = None
knl_utop_prep = None
knl_utop_set = None
knl_set_eflag = None
ncov = None
ncon = None
lgdet = None
def U_to_P(params, G, U, P, Pout=None, iter_max=None):

    s = G.slices
    sh = G.shapes

    # Set some default parameters, but allow overrides
    if iter_max is None:
        if 'invert_iter_max' in params:
            iter_max = params['invert_iter_max']
        else:
            iter_max = 8

    if 'invert_err_tol' in params:
        err_tol = params['invert_err_tol']
    else:
        err_tol = 1.e-8

    if 'invert_iter_delta' in params:
        delta = params['invert_iter_delta']
    else:
        delta = 1.e-5

    if 'gamma_max' not in params:
        params['gamma_max'] = 25

    # Don't overwrite old memory by default, just allow using old values
    # Caller can pass new/old memory but is responsible that it contain logical values if we don't update it
    if Pout is None:
        Pout = P.copy()

    # Update the primitive B-fields
    G.vecdivbygeom(params['queue'], u=U[s.B3VEC], g=G.gdet_d[Loci.CENT.value], out=Pout[s.B3VEC])

    # Cached constant quantities
    global ncov, ncon, lgdet
    if ncov is None:
        # For later on
        ncov = cl_array.zeros(params['queue'], sh.grid_vector, dtype=np.float64)
        G.timesgeom(params['queue'], u=cl_array.empty_like(ncov[0]).fill(1.0),
                    g=-G.lapse_d[Loci.CENT.value], out=ncov[0])
        ncon = G.raise_grid(ncov)
        lgdet = G.lapse_d[Loci.CENT.value] / G.gdet_d[Loci.CENT.value]

    # Eflag will indicate inversion failures
    # Define a generic kernel so we can split out flagging in the future
    # Use it to catch negative density early on
    eflag = cl_array.zeros(params['queue'], sh.grid_scalar, dtype=np.int32)
    global knl_set_eflag
    if knl_set_eflag is None:
        code = add_ghosts("""eflag[i,j,k] = if(var[i,j,k] < 0, flag, eflag[i,j,k])""")
        knl_set_eflag = lp.make_kernel(sh.isl_grid_scalar, code,
                                       [*scalarArrayArgs("eflag", dtype=np.int32),
                                        *scalarArrayArgs("var"), lp.ValueArg("flag", dtype=np.int32),
                                        ...], default_offset=lp.auto)
        knl_set_eflag = tune_grid_kernel(knl_set_eflag, sh.bulk_scalar, ng=G.NG)

    evt, _ = knl_set_eflag(params['queue'], var=U[s.RHO], eflag=eflag, flag=-100)

    # Convert from conserved variables to four-vectors
    Bcon = cl_array.zeros(params['queue'], sh.grid_vector, dtype=np.float64)
    G.vectimesgeom(params['queue'], u=U[s.B3VEC], g=lgdet, out=Bcon[1:])

    Qcov = cl_array.empty_like(Bcon)
    G.timesgeom(params['queue'], u=(U[s.UU] - U[s.RHO]), g=lgdet, out=Qcov[0])
    G.vectimesgeom(params['queue'], u=U[s.U3VEC], g=lgdet, out=Qcov[1:])

    Bcov = G.lower_grid(Bcon)
    Qcon = G.raise_grid(Qcov)

    # This will have fringes of zeros still!
    Bsq = G.dot(Bcon, Bcov)
    QdB = G.dot(Bcon, Qcov)
    Qdotn = G.dot(Qcon, ncov)
    Qsq = G.dot(Qcon, Qcov)

    Qtsq = Qsq + Qdotn**2

    Qtcon = cl_array.empty_like(Qcon)
    for i in range(4):
        Qtcon[i] = Qcon[i] + ncon[i] * Qdotn

    # Set up eqn for W', the energy density
    D = cl_array.zeros_like(Qsq)
    G.timesgeom(params['queue'], u=U[s.RHO], g=lgdet, out=D)
    Ep = -Qdotn - D

    del Bcov, Qcon, Qcov

    # Numerical rootfinding
    # Take guesses from primitives
    Wp = Wp_func(params, G, P, Loci.CENT, eflag)
    # Trap on any failures so far if debugging.  They're very rare.
    if 'debug' in params and params['debug']:
        if np.any(eflag.get()[s.bulk] != 0):
            raise ValueError("Unexpected flag set!")

    # Step around the guess & evaluate errors
    h = delta * Wp  # TODO stable enough?  Need fancy subtraction from iharm3d?
    errp = err_eqn(params, G, Bsq, D, Ep, QdB, Qtsq, Wp + h, eflag)
    err =  err_eqn(params, G, Bsq, D, Ep, QdB, Qtsq, Wp, eflag)
    errm = err_eqn(params, G, Bsq, D, Ep, QdB, Qtsq, Wp - h, eflag)

    # Preserve Wp/err before updating them below
    Wp1 = Wp.copy()
    err1 = err.copy()

    global knl_utop_prep
    if knl_utop_prep is None:
        # TODO keep an accumulator here to avoid that costly any() call?
        code = add_ghosts("""
        # TODO put error/prep in here, not calling below

        # Attempt a Halley/Muller/Bailey/Press step
        dedW := (errp[i,j,k] - errm[i,j,k]) / (2 * h[i,j,k])
        dedW2 := (errp[i,j,k] - 2.*err[i,j,k] + errm[i,j,k]) / (h[i,j,k]**2)

        # Limit size of 2nd derivative correction
        # Loopy trick common in HARM: define intermediate variables (xt, xt2, xt3...) to impose clipping
        # This allows assignments or substitutions while keeping dependencies straight
        ft := 0.5*err[i,j,k]*dedW2/(dedW**2)
        ft2 := if(ft > 0.3, 0.3, ft)
        f := if(ft2 < -0.3, -0.3, ft2)

        # Limit size of step
        dWt := -err[i,j,k] / dedW / (1. - f)
        dWt2 := if(dWt < -0.5*Wp[i,j,k], -0.5*Wp[i,j,k], dWt)
        dW := if(dWt2 > 2.0*Wp[i,j,k], 2.0*Wp[i,j,k], dWt2)

        Wp[i,j,k] = Wp[i,j,k] + dW {id=wp}

        # Guarantee we take one step in every bulk zone
        stop_flag[i,j,k] = 0
        # This would avoid the step where there's convergence, but would require taking 2*dW above or similar
        #stop_flag[i,j,k] = (fabs(dW / Wp[i,j,k]) < err_tol) + \
        #                   (fabs(err[i,j,k] / Wp[i,j,k]) < err_tol) {dep=wp,nosync=wp}
        """)
        knl_utop_prep = lp.make_kernel(sh.isl_grid_scalar, code,
                                       [*scalarArrayArgs("err", "errp", "errm", "h", "Wp"),
                                        *scalarArrayArgs("stop_flag", dtype=np.int8), ...],
                                       assumptions=sh.assume_grid, seq_dependencies=True)
        knl_utop_prep = lp.fix_parameters(knl_utop_prep, err_tol=err_tol)
        knl_utop_prep = tune_grid_kernel(knl_utop_prep, sh.bulk_scalar, ng=G.NG)
        print("Compiled utop_prep")

    # Fill stop_flag with 1 so we don't have to worry about ghost zones taking steps
    stop_flag = cl_array.empty(params['queue'], sh.grid_scalar, dtype=np.int8).fill(1)

    evt, _ = knl_utop_prep(params['queue'], err=err, errp=errp, errm=errm, h=h,
                           Wp=Wp, stop_flag=stop_flag)
    evt.wait()

    err_eqn(params, G, Bsq, D, Ep, QdB, Qtsq, Wp, eflag, out=err)

    # Iteration kernel for 1Dw solver
    global knl_utop_iter
    if knl_utop_iter is None:
        code = add_ghosts("""
        # Evaluate whether we need to do any of this
        <> go = not(stop_flag[i,j,k]) {id=insn_go}

        # Normal secant increment is dW. Limit guess to between 0.5 and 2 times current value
        dWt := (Wp1[i,j,k] - Wp[i,j,k]) * err[i,j,k] / (err[i,j,k] - err1[i,j,k])
        dWt2 := if(dWt < -0.5*Wp[i,j,k], -0.5*Wp[i,j,k], dWt)
        <> dW = if(dWt2 > 2.0*Wp[i,j,k], 2.0*Wp[i,j,k], dWt2) {id=dw,if=go}

        # Preserve last values, after use but before any changes
        Wp1[i,j,k] = Wp[i,j,k] {id=wp1,nosync=dw,if=go}
        err1[i,j,k] = err[i,j,k] {nosync=dw,if=go}

        # Update Wp.  Err will be updated outside kernel
        Wp[i,j,k] = Wp[i,j,k] + dW {id=wp,nosync=dw:wp1,if=go}

        # Set flag not to continue in zones that have converged
        stop_flag[i,j,k] = if(fabs(dW / Wp[i,j,k]) < err_tol, 1, stop_flag[i,j,k]) {nosync=dw:wp:insn_go,if=go}

        # For the future, when we've defined err_eqn for loopy kernels
        # err = err_eqn(Bsq, D, Ep, QdB, Qtsq, Wp, gam, gamma_max, eflag) {if=go}
        # stop_flag[i,j,k] = stop_flag[i,j,k] + (fabs(err[] / Wp[]) < err_tol) {if=go}
        """)
        knl_utop_iter = lp.make_kernel(sh.isl_grid_scalar, code,
                                       [*scalarArrayArgs("Wp", "Wp1", "err", "err1"),
                                        *scalarArrayArgs("stop_flag", dtype=np.int8), ...],
                                       assumptions=sh.assume_grid, default_offset=lp.auto,
                                       seq_dependencies=True)
        knl_utop_iter = lp.fix_parameters(knl_utop_iter, err_tol=err_tol)
        knl_utop_iter = tune_grid_kernel(knl_utop_iter, sh.bulk_scalar, ng=G.NG)
        print("Compiled utop_iter")

    # Iterate at least once to set new values from first step
    # TODO Needed now we set Wp, err1 right?
    for niter in range(iter_max):
        #print("U_to_P iter")
        evt, _ = knl_utop_iter(params['queue'], Wp=Wp, Wp1=Wp1, err=err, err1=err1, stop_flag=stop_flag)
        err = err_eqn(params, G, Bsq, D, Ep, QdB, Qtsq, Wp, eflag)
        # TODO there may be better/faster if/reduction statements here...
        stop_flag |= (clm.fabs(err / Wp) < err_tol)
        if cl_array.min(stop_flag) >= 1:
            break

    # If secant method failed to converge, do not set primitives other than B
    eflag += (stop_flag == 0)

    del Wp1, err, err1, stop_flag

    # Find utsq, gamma, rho0 from Wp
    gamma = gamma_func(params, G, Bsq, D, QdB, Qtsq, Wp, eflag)

    if 'debug' in params and params['debug']:
        if np.any(gamma.get()[s.bulk] < 1.):
            raise ValueError("gamma < 1 failure!")

    # Find the scalars
    global knl_utop_set
    if knl_utop_set is None:
        code = add_ghosts(replace_prim_names("""
        rho0 := D[i,j,k] / gamma[i,j,k]
        W := Wp[i,j,k] + D[i,j,k]
        w := W / (gamma[i,j,k]**2)
        pres := (w - rho0) * (gam - 1.) / gam
        u := w - (rho0 + pres)

        # Set flag if prims are < 0
        eflag[i,j,k] = if((u < 0)*(rho0 < 0), 8, if(u < 0, 7, if(rho0 < 0, 6, eflag[i,j,k]))) {id=ef}

        # Don't update flagged primitives (necessary?  Could skip the branch if fixup does ok)
        <> set = not(eflag[i,j,k]) {dep=ef,nosync=ef}

        P[RHO,i,j,k] = rho0 {if=set}
        P[UU,i,j,k] = u     {if=set}
        P[U1,i,j,k] = (gamma[i,j,k] / (W + Bsq[i,j,k])) * (Qtcon[1,i,j,k] + QdB[i,j,k] * Bcon[1,i,j,k] / W) {if=set}
        P[U2,i,j,k] = (gamma[i,j,k] / (W + Bsq[i,j,k])) * (Qtcon[2,i,j,k] + QdB[i,j,k] * Bcon[2,i,j,k] / W) {if=set}
        P[U3,i,j,k] = (gamma[i,j,k] / (W + Bsq[i,j,k])) * (Qtcon[3,i,j,k] + QdB[i,j,k] * Bcon[3,i,j,k] / W) {if=set}
        """))
        if 'electrons' in params and params['electrons']:
            code += add_ghosts("""
            P[KEL,i,j,k] = U[KEL,i,j,k]/U[RHO,i,j,k]
            P[KTOT,i,j,k] = U[KTOT,i,j,k]/U[RHO,i,j,k]
            """)
        knl_utop_set = lp.make_kernel(sh.isl_grid_scalar, code,
                                       [*primsArrayArgs("P"),
                                        *vecArrayArgs("Qtcon", "Bcon"),
                                        *scalarArrayArgs("D", "gamma", "Wp", "Bsq", "QdB"),
                                        *scalarArrayArgs("eflag", dtype=np.int32), ...],
                                       assumptions=sh.assume_grid, default_offset=lp.auto)
        knl_utop_set = lp.fix_parameters(knl_utop_set, gam=params['gam'], nprim=params['n_prims'], ndim=4)
        knl_utop_set = tune_grid_kernel(knl_utop_set, sh.bulk_scalar, ng=G.NG)
        print("Compiled utop_set")

    evt, _ = knl_utop_set(params['queue'], P=Pout, Qtcon=Qtcon, Bcon=Bcon,
                          D=D, gamma=gamma, Wp=Wp, Bsq=Bsq, QdB=QdB, eflag=eflag)
    evt.wait()
    del Qtcon, Bcon, D, gamma, Wp, Bsq, QdB

    # Trap on flags early in test problems
    if 'debug' in params and params['debug']:
        n_nonzero = np.count_nonzero(eflag.get())
        if n_nonzero > 0:
            print("Nonzero eflag in bulk: {}\nFlags: {}".format(n_nonzero, np.argwhere(eflag.get() != 0)))

    return Pout, eflag


knl_err_eqn = None
def err_eqn(params, G, Bsq,  D,  Ep,  QdB,  Qtsq, Wp, eflag, out=None):
    sh = G.shapes

    gamma = gamma_func(params, G, Bsq, D, QdB, Qtsq, Wp, eflag)

    global knl_err_eqn
    if knl_err_eqn is None:
        code = add_ghosts("""
        W := Wp[i,j,k] + D[i,j,k]
        w := W / (gamma[i,j,k]**2)
        rho0 := D[i,j,k] / gamma[i,j,k]
        pres := (w - rho0) * (gam - 1.) / gam

        err[i,j,k] = -Ep[i,j,k] + Wp[i,j,k] - pres + 0.5*Bsq[i,j,k] + \
                        0.5*(Bsq[i,j,k] * Qtsq[i,j,k] - QdB[i,j,k]**2)/((Bsq[i,j,k] + W)**2)
        """)
        knl_err_eqn = lp.make_kernel(sh.isl_grid_scalar, code,
                                        [*scalarArrayArgs("gamma", "Bsq", "D", "Ep", "QdB", "Qtsq", "Wp", "err"),
                                         *scalarArrayArgs("eflag", dtype=np.int32),
                                         ...],
                                     assumptions=sh.assume_grid, default_offset=lp.auto)
        knl_err_eqn = lp.fix_parameters(knl_err_eqn, nprim=params['n_prims'], gam=params['gam'],
                                        gamma_max=params['gamma_max'])
        knl_err_eqn = tune_grid_kernel(knl_err_eqn, sh.bulk_scalar, ng=G.NG)

    if out is None:
        out = cl_array.zeros_like(Bsq)

    evt, _ = knl_err_eqn(params['queue'], Bsq=Bsq, D=D, Ep=Ep, QdB=QdB, Qtsq=Qtsq, Wp=Wp, gamma=gamma,
                         err=out, eflag=eflag)

    return out


knl_gamma_func = None
def gamma_func(params, G, Bsq,  D,  QdB,  Qtsq,  Wp, eflag, out=None):
    sh = G.shapes

    global knl_gamma_func
    if knl_gamma_func is None:
        code = add_ghosts("""
        W := D[i,j,k] + Wp[i,j,k]
        WB := W + Bsq[i,j,k]
        # This is basically inversion of eq. A7 of MM
        <> utsq = -((W + WB) * QdB[i,j,k]**2 + W**2 * Qtsq[i,j,k]) / \
                    (QdB[i,j,k]**2 * (W + WB) + W**2 * (Qtsq[i,j,k] - WB**2))

        # Catch utsq < 0 and record it
        cond := ((utsq < 0) + (utsq > 1.e3 * gamma_max ** 2))
        eflag[i,j,k] = if(cond, 2, eflag[i,j,k])

        gamma[i,j,k] = sqrt(1. + fabs(utsq))
        """)
        knl_gamma_func = lp.make_kernel(sh.isl_grid_scalar, code,
                                        [*scalarArrayArgs("Bsq", "D", "QdB", "Qtsq", "Wp", "gamma"),
                                         *scalarArrayArgs("eflag", dtype=np.int32),
                                         ...],
                                     assumptions=sh.assume_grid, default_offset=lp.auto)
        knl_gamma_func = lp.fix_parameters(knl_gamma_func, gamma_max=params['gamma_max'])
        knl_gamma_func = tune_grid_kernel(knl_gamma_func, sh.bulk_scalar, ng=G.NG)
        print("Compiled gamma_func")

    if out is None:
        out = cl_array.zeros_like(Bsq)

    evt, _ = knl_gamma_func(params['queue'], Bsq=Bsq, D=D, QdB=QdB, Qtsq=Qtsq, Wp=Wp, gamma=out, eflag=eflag)

    return out


knl_Wp_func = None
def Wp_func(params, G, P, loc, eflag, out=None):
    s = G.slices
    sh = G.shapes

    # Again, vectors are done full-grid
    utcon = cl_array.empty(params['queue'], sh.grid_vector, dtype=np.float64)
    utcon[0] = 0
    utcon[1:] = P[s.U3VEC]

    utcov = G.lower_grid(utcon, loc)
    utsq = G.dot(utcon, utcov)

    global knl_Wp_func
    if knl_Wp_func is None:
        code = add_ghosts(replace_prim_names("""
        cond1 := ((utsq_in[i,j,k] < 0.) * (abs(utsq_in[i,j,k]) < 1.e-13))
        utsq1 := if(cond1, fabs(utsq_in[i,j,k]), utsq_in[i,j,k])

        # Catch utsq < 0 and record it
        cond2 := ((utsq1 < 0) + (utsq1 > 1.e3 * gamma_max ** 2))
        utsq := if(cond2, (P[RHO,i,j,k] + P[UU,i,j,k]), utsq1)
        eflag[i,j,k] = if(cond2, 2, eflag[i,j,k])

        gamma := sqrt(1. + fabs(utsq))
        Wp[i,j,k] = (P[RHO,i,j,k] + P[UU,i,j,k] + (gam - 1.) * P[UU,i,j,k]) * gamma ** 2 - P[RHO,i,j,k] * gamma
        """))
        knl_Wp_func = lp.make_kernel(sh.isl_grid_scalar, code,
                                     [*primsArrayArgs("P"), *scalarArrayArgs("utsq_in", "Wp"),
                                      *scalarArrayArgs("eflag", dtype=np.int32),
                                      ...],
                                     assumptions=sh.assume_grid, default_offset=lp.auto)
        knl_Wp_func = lp.fix_parameters(knl_Wp_func, nprim=params['n_prims'], gam=params['gam'],
                                        gamma_max=params['gamma_max'])
        knl_Wp_func = tune_grid_kernel(knl_Wp_func, sh.bulk_scalar, ng=G.NG)
        print("Compiled Wp_func")

    if out is None:
        out = cl_array.zeros_like(utsq)

    evt, _ = knl_Wp_func(params['queue'], P=P, utsq_in=utsq, Wp=out, eflag=eflag)

    return out

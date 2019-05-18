# Physics functions

import numpy as np

import pyopencl.array as cl_array
import pyopencl.clmath as clm
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from loopy_tools import *

from defs import Loci

knl_prim_to_flux = None
knl_Tmhd_vec = None
knl_ucon_calc = None
knl_mhd_vchar = None

# Temporary variables I don't want to re-declare
# TODO objects. Memoization/instantiation -> objects, stupid
bsq = None
gammaG = None

# Stash pieces of the geometry as temps. See above objects comment
g3 = None
gcon1_d = None
gcon2_d = None
gcon3_d = None

def Tmhd_vec(params, G, P, D, dir, out=None):
    """Compute one column the MHD stress-energy tensor with first index up, second index down.
    Note a factor of sqrt(4 pi) is absorbed into the definition of b.
    """
    # TODO sort these out/unify with T_mixed in ana.
    # phys.py and variables.py can actually share most stuff...
    sh = G.shapes

    if out is None:
        out = cl_array.empty(params['queue'], sh.grid_vector, dtype=np.float64)

    global knl_Tmhd_vec
    if knl_Tmhd_vec is None:
        code = replace_prim_names("""
        <> bsq = sum(nu, bcon[nu, i, j, k] * bcov[nu, i, j, k])
        w := P[RHO,i,j,k] + gam * P[UU,i,j,k] + bsq
        T[mu,i,j,k] = w * ucon[dir,i,j,k] * ucov[mu,i,j,k] - bcon[dir,i,j,k] * bcov[mu,i,j,k] \
                        + if(mu == dir, (gam - 1) * P[UU,i,j,k] + 0.5 * bsq, 0)
        """)
        knl_Tmhd_vec = lp.make_kernel("[dir, ndim, n1, n2, n3] -> " + sh.isl_grid_tensor, code,
                                      [lp.ValueArg("gam"), *primsArrayArgs("P", ghosts=False),
                                       *vecArrayArgs("ucon", "ucov", "bcon", "bcov", ghosts=False),
                                       ...],
                                      assumptions=sh.assume_grid + "and 0 <= dir < ndim",
                                      default_offset=lp.auto)
        knl_Tmhd_vec = lp.fix_parameters(knl_Tmhd_vec, gam=params['gam'], nprim=params['n_prims'], ndim=4)
        knl_Tmhd_vec = tune_grid_kernel(knl_Tmhd_vec, sh.grid_vector)
        knl_Tmhd_vec = lp.tag_inames(knl_Tmhd_vec, "nu:unr,mu:unr")
        print("Compiled Tmhd_vec")

    evt, _ = knl_Tmhd_vec(params['queue'], P=P, ucon=D['ucon'], ucov=D['ucov'],
                          bcon=D['bcon'], bcov=D['bcov'], T=out, dir=dir)

    return out


def prim_to_flux(params, G, P, D=None, dir=0, loc=Loci.CENT, out=None):
    """Calculate fluxes of conserved varibles in direction dir, or if dir=0 the variables themselves"""
    sh = G.shapes

    if out is None:
        out = cl_array.empty_like(P)
    if D is None:
        D = get_state(params, G, P, loc)

    global knl_prim_to_flux
    if knl_prim_to_flux is None:
        code = replace_prim_names("""
        out[RHO,i,j,k] = P[RHO,i,j,k] * ucon[dir,i,j,k] * gdet[i,j]
        out[UU,i,j,k]  = (T[0,i,j,k] + P[RHO,i,j,k] * ucon[dir,i,j,k]) * gdet[i,j]
        out[U1,i,j,k]  = T[1,i,j,k] * gdet[i,j]
        out[U2,i,j,k]  = T[2,i,j,k] * gdet[i,j]
        out[U3,i,j,k]  = T[3,i,j,k] * gdet[i,j]
        out[B1,i,j,k]  = (bcon[1,i,j,k] * ucon[dir,i,j,k] - bcon[dir,i,j,k] * ucon[1,i,j,k]) * gdet[i,j]
        out[B2,i,j,k]  = (bcon[2,i,j,k] * ucon[dir,i,j,k] - bcon[dir,i,j,k] * ucon[2,i,j,k]) * gdet[i,j]
        out[B3,i,j,k]  = (bcon[3,i,j,k] * ucon[dir,i,j,k] - bcon[dir,i,j,k] * ucon[3,i,j,k]) * gdet[i,j]
        """)
        if 'electrons' in params and params['electrons']:
            code += replace_prim_names("""
            out[KEL,i,j,k] = P[RHO,i,j,k] * ucon[dir,i,j,k] * gdet[i,j] * P[KEL,i,j,k]
            out[KTOT,i,j,k] = P[RHO,i,j,k] * ucon[dir,i,j,k] * gdet[i,j] * P[KTOT,i,j,k]
            """)
        # TODO also passives
        knl_prim_to_flux = lp.make_kernel("[dir, ndim, n1, n2, n3] -> " + sh.isl_grid_scalar, code,
                                          [*primsArrayArgs("P", "out", ghosts=False),
                                           *vecArrayArgs("T", "ucon", "bcon", ghosts=False),
                                           *gscalarArrayArgs("gdet", ghosts=False), ...],
                                          assumptions=sh.assume_grid + "and 0 <= dir < ndim",
                                          default_offset=lp.auto)
        knl_prim_to_flux = lp.fix_parameters(knl_prim_to_flux, nprim=params['n_prims'], ndim=4)
        # TODO keep k because of the geom argument
        knl_prim_to_flux = tune_grid_kernel(knl_prim_to_flux, sh.grid_scalar)
        print("Compiled prim_to_flux")

    evt, _ = knl_prim_to_flux(params['queue'], P=P, T=Tmhd_vec(params, G, P, D, dir),
                              gdet=G.gdet_d[loc.value], ucon=D['ucon'], bcon=D['bcon'],
                              dir=dir, out=out)
    if 'profile' in params and params['profile']:
        evt.wait()

    return out


def mhd_gamma_calc(queue, G, P, loc=Loci.CENT, out=None):
    """Find relativistic gamma-factor w.r.t. normal observer"""
    s = G.slices
    sh = G.shapes

    global g3
    if g3 is None:
        g3 = cl_array.to_device(queue, G.gcov[loc.value, 1:, 1:].copy())

    if out is None:
        out = cl_array.empty(queue, sh.grid_scalar, dtype=np.float64)

    evt, _ = G.dot2geom2(queue, g=g3, u=P[s.U3VEC], v=P[s.U3VEC], out=out)
    out = clm.sqrt(1. + out)
    return out


def ucon_calc(params, G, P, loc, out=None):
    """Find contravariant fluid four-velocity"""
    sh = G.shapes

    if out is None:
        out = cl_array.empty(params['queue'], sh.grid_vector, dtype=np.float64)

    global knl_ucon_calc
    if knl_ucon_calc is None:
        code = replace_prim_names("""
        # Replicate mhd_gamma_calc here for speed.  Not ideal :(
        <> qsq =  gcov[1,1,i,j]*P[U1,i,j,k]**2 \
                + gcov[2,2,i,j]*P[U2,i,j,k]**2 \
                + gcov[3,3,i,j]*P[U3,i,j,k]**2 \
                + 2.*(gcov[1,2,i,j]*P[U1,i,j,k]*P[U2,i,j,k] \
                    + gcov[1,3,i,j]*P[U1,i,j,k]*P[U3,i,j,k] \
                    + gcov[2,3,i,j]*P[U2,i,j,k]*P[U3,i,j,k])
        gamma := sqrt(1. + qsq)

        ucon[0,i,j,k] = gamma / lapse[i,j]
        ucon[1,i,j,k] = P[U1,i,j,k] - gamma * lapse[i,j] * gcon[1,i,j]
        ucon[2,i,j,k] = P[U2,i,j,k] - gamma * lapse[i,j] * gcon[2,i,j]
        ucon[3,i,j,k] = P[U3,i,j,k] - gamma * lapse[i,j] * gcon[3,i,j]
        """)
        knl_ucon_calc = lp.make_kernel(sh.isl_grid_scalar, code,
                                       [*primsArrayArgs("P", ghosts=False),
                                        *vecArrayArgs("ucon", ghosts=False),
                                        *gvectorArrayArgs("gcon", ghosts=False),
                                        *gtensorArrayArgs("gcov", ghosts=False), ...],
                                       assumptions=sh.assume_grid, default_offset=lp.auto)
        knl_ucon_calc = lp.fix_parameters(knl_ucon_calc, nprim=params['n_prims'], ndim=4)
        knl_ucon_calc = tune_grid_kernel(knl_ucon_calc, sh.grid_scalar)
        print("Compiled ucon_calc")

    evt, _ = knl_ucon_calc(params['queue'], P=P, lapse=G.lapse_d[loc.value],
                           gcov=G.gcov_d[loc.value], gcon=G.gcon_d[loc.value, 0],  # Different geometries! See code
                           ucon=out)
    if 'profile' in params and params['profile']:
        evt.wait()

    return out


knl_bcon_calc = None
def bcon_calc(params, G, P, ucon, ucov, out=None):
    """Calculate magnetic field four-vector"""
    sh = G.shapes

    if out is None:
        out = cl_array.empty_like(ucon)

    global knl_bcon_calc
    if knl_bcon_calc is None:
        code = replace_prim_names("""
        bcon[0,i,j,k] = P[B1,i,j,k]*ucov[1,i,j,k] + P[B2,i,j,k]*ucov[2,i,j,k] + P[B3,i,j,k]*ucov[3,i,j,k] {id=bcon0}
        bcon[1+mu,i,j,k] = (P[B1+mu,i,j,k] + bcon[0,i,j,k] * ucon[1+mu,i,j,k]) / \
                            ucon[0,i,j,k] {id=bconv,dep=bcon0,nosync=bcon0}
        """)
        knl_bcon_calc = lp.make_kernel(sh.isl_grid_3vector, code,
                                  [*primsArrayArgs("P", ghosts=False),
                                   *vecArrayArgs("ucon", "ucov", "bcon", ghosts=False), ...],
                                  assumptions=sh.assume_grid, default_offset=lp.auto)
        knl_bcon_calc = lp.fix_parameters(knl_bcon_calc, nprim=params['n_prims'], ndim=4)
        knl_bcon_calc = tune_grid_kernel(knl_bcon_calc, sh.grid_scalar)
        print("Compiled bcon_calc")

    evt, _ = knl_bcon_calc(params['queue'], P=P, ucon=ucon, ucov=ucov, bcon=out)
    if 'profile' in params and params['profile']:
        evt.wait()

    return out


def get_state(params, G, P, loc=Loci.CENT, out=None):
    """Calculate ucon, ucov, bcon, bcov from primitive variables
    Returns a dict of state variables
    """
    # TODO make this a fusion of kernels?  Components are needed in places
    if out is None:
        out = {}
        out['ucon'] = cl_array.empty(params['queue'], G.shapes.grid_vector, dtype=np.float64)
        out['ucov'] = cl_array.empty_like(out['ucon'])
        out['bcon'] = cl_array.empty_like(out['ucon'])
        out['bcov'] = cl_array.empty_like(out['ucon'])
    ucon_calc(params, G, P, loc, out=out['ucon'])
    G.lower_grid(out['ucon'], loc, out=out['ucov'])
    bcon_calc(params, G, P, out['ucon'], out['ucov'], out=out['bcon'])
    G.lower_grid(out['bcon'], loc, out=out['bcov'])
    return out

Acov = None
Acon = None
Bcov = None
Bcon = None
def mhd_vchar(params, G, P, D, loc, dir, vmax_out=None, vmin_out=None):
    """Calculate components of magnetosonic velocity from primitive variables"""
    sh = G.shapes

    # TODO these can almost certainly be calculated on the fly. Just need geometry
    # Careful Acov and *con change with dir/loc arguments
    global Acov, Acon, Bcov, Bcon
    if Acov is None:
        Acov = cl_array.empty(params['queue'], sh.grid_vector, dtype=np.float64)
        Bcov = cl_array.zeros_like(Acov)
        Bcov[0] = 1
        Acon = cl_array.empty_like(Acov)
        Bcon = cl_array.empty_like(Acov)

    # Acov needs to change with dir in call
    Acov.fill(0)
    Acov[dir].fill(1)

    G.raise_grid(Acov, loc, out=Acon)
    G.raise_grid(Bcov, loc, out=Bcon)

    # Find fast magnetosonic speed
    global knl_mhd_vchar
    if knl_mhd_vchar is None:
        code = replace_prim_names("""
        <> bsq = simul_reduce(sum, mu, bcon[mu,i,j,k] * bcov[mu,i,j,k])
        <> Asq = simul_reduce(sum, mu, Acon[mu,i,j,k] * Acov[mu,i,j,k])
        <> Bsq = simul_reduce(sum, mu, Bcon[mu,i,j,k] * Bcov[mu,i,j,k])
        <> AB = simul_reduce(sum, mu, Acon[mu,i,j,k] * Bcov[mu,i,j,k])
        <> Au = simul_reduce(sum, mu, Acov[mu,i,j,k] * ucon[mu,i,j,k])
        <> Bu = simul_reduce(sum, mu, Bcov[mu,i,j,k] * ucon[mu,i,j,k])

        ef := fabs(P[RHO,i,j,k]) + gam * fabs(P[UU,i,j,k])
        ee := (bsq + ef)
        <> va2 = bsq / ee
        <> cs2 = gam * (gam - 1.) * fabs(P[UU,i,j,k]) / ef

        # Keep two temps to keep loopy from having to compile complete spaghetti
        cms21 := cs2 + va2 - cs2 * va2
        cms22 := if(cms21 > 0, cms21, 0)
        <> cms2 = if(cms22 > 1, 1, cms22)

        A := Bu**2 - (Bsq + Bu**2) * cms2
        B := 2. * (Au * Bu - (AB + Au * Bu) * cms2)
        C := Au**2 - (Asq + Au**2) * cms2

        <> discr = sqrt(if(B**2 - 4.*A*C > 0, B**2 - 4.*A*C, 0))

        vp := -(-B + discr) / (2. * A)
        vm := -(-B - discr) / (2. * A)

        vmax[i,j,k] = if(vp > vm, vp, vm)
        vmin[i,j,k] = if(vp > vm, vm, vp)
        """)
        knl_mhd_vchar = lp.make_kernel(sh.isl_grid_vector, code,
                                       [*primsArrayArgs("P", ghosts=False), ...],
                                       assumptions=sh.assume_grid, default_offset=lp.auto)

        knl_mhd_vchar = lp.fix_parameters(knl_mhd_vchar, gam=params['gam'], nprim=params['n_prims'])
        knl_mhd_vchar = tune_grid_kernel(knl_mhd_vchar, sh.grid_vector)
        knl_mhd_vchar = lp.tag_inames(knl_mhd_vchar, "mu:unr")
        print("Compiled mhd_vchar")

    if vmax_out is None:
        vmax_out = cl_array.empty(params['queue'], sh.grid_scalar, dtype=np.float64)
    if vmin_out is None:
        vmin_out = cl_array.empty(params['queue'], sh.grid_scalar, dtype=np.float64)

    evt, _ = knl_mhd_vchar(params['queue'], P=P, Acon=Acon, Acov=Acov, Bcon=Bcon, Bcov=Bcov,
                           ucon=D['ucon'], bcon=D['bcon'], bcov=D['bcov'],
                           vmax=vmax_out, vmin=vmin_out)
    if 'profile' in params and params['profile']:
        evt.wait()

    return vmax_out, vmin_out


def get_fluid_source(params, G, P, D, out=None):
    """Calculate a small fluid source term, added to conserved variables for stability"""
    s = G.slices
    sh = G.shapes

    # T the old fashioned way: TODO Tmhd_full...
    T = cl_array.empty(params['queue'], sh.grid_tensor, dtype=np.float64)
    for mu in range(4):
        Tmhd_vec(params, G, P, D, mu, out=T[mu])

    if out is None:
        out = cl_array.empty_like(P)

    global gcon1_d, gcon2_d, gcon3_d
    if gcon1_d is None:
        gcon1_d = cl_array.to_device(params['queue'], (G.conn[:, :, 1, :, :]*G.gdet[Loci.CENT.value]).copy())
        gcon2_d = cl_array.to_device(params['queue'], (G.conn[:, :, 2, :, :]*G.gdet[Loci.CENT.value]).copy())
        gcon3_d = cl_array.to_device(params['queue'], (G.conn[:, :, 3, :, :]*G.gdet[Loci.CENT.value]).copy())

    # Contract mhd stress tensor with connection
    evt, _ = G.dot2D2geom(params['queue'], u=T, g=gcon1_d, out=out[s.U1])
    evt, _ = G.dot2D2geom(params['queue'], u=T, g=gcon2_d, out=out[s.U2])
    evt, _ = G.dot2D2geom(params['queue'], u=T, g=gcon3_d, out=out[s.U2])

    if 'profile' in params and params['profile']:
        evt.wait()

    return out

    # # Add a small "wind" source term in RHO,UU
    # # Stolen shamelessly from iharm2d_v3
    # if params['wind_term']:
    #     # need coordinates to evaluate particle addtn rate
    #     X = G.coord_bulk(Loci.CENT)
    #     r, th, _ = G.ks_coord(X)
    #     cth = np.cos(th)
    #
    #     # here is the rate at which we're adding particles
    #     # this function is designed to concentrate effect in the
    #     # funnel in black hole evolutions
    #     drhopdt = 2.e-4*cth**4/(1. + r**2)**2
    #
    #     dP[RHO] = drhopdt
    #
    #     Tp = 10.   # temp, in units of c^2, of new plasma
    #     dP[UU] = drhopdt*Tp*3.
    #
    #     # Leave P[U1,2,3]=0 to add in particles in normal observer frame
    #     # Likewise leave P[BN]=0
    #
    #
    #     # add in plasma to the T^t_a component of the stress-energy tensor
    #     # notice that U already contains a factor of sqrt-g
    #     dD = get_state_vec(G, dP, Loci.CENT)
    #     ddU = prim_to_flux(G, dP, dD, 0, Loci.CENT)
    #
    #     (*dU)[ip] += U[ip]
    # Remember this from above
    # for p in range(params['n_prims']):
    #     G.timesgeom(params['queue'], u=out[p], g=G.gdet_d[Loci.CENT.value], out=out[p])

# Physical boundary conditions

import numpy as np

from phys import ucon_calc, mhd_gamma_calc
from defs import Loci


user_bound_l = None
user_bound_r = None
def register_user_bound(func_left=None, func_right=None):
    global user_bound_l, user_bound_r
    if func_left is not None:
        user_bound_l = func_left
    if func_right is not None:
        user_bound_r = func_right


def set_bounds(params, G, P, pflag=None):
    NG = G.NG
    s = G.slices

    ################################## X1 BOUNDARY ##########################################

    if G.global_start[0] == 0:
        slc_to = (s.ghostl, s.b, s.b)
        if params["x1l_bound"] == "outflow":
            slc_from = (s.boundl_o, s.b, s.b)
            P[s.allv + slc_to] = P[s.allv + slc_from]
            if pflag is not None:
                pflag[slc_to] = pflag[slc_from]

            rescale = G.gdet[Loci.CENT.value, NG, :] / G.gdet[Loci.CENT.value, s.ghostl, :]
            P[s.B1 + slc_to] *= rescale[:, s.b, None]
            P[s.B2 + slc_to] *= rescale[:, s.b, None]
            P[s.B3 + slc_to] *= rescale[:, s.b, None]

        elif params["x1l_bound"] == "periodic" and G.global_stop[0] == G.NTOT[1]:
            if G.global_stop[0] == G.NTOT[1]:
                slc_from = (s.boundr, s.b, s.b)
                P[s.allv + slc_to] = P[s.allv + slc_from]
                if pflag is not None:
                    pflag[slc_to] = pflag[slc_from]

        elif params['x1l_bound'] == "user":
            P[s.allv + slc_to] = user_bound_l(params, G, P, slc=slc_to)
            if pflag is not None:
                pflag[slc_to] = 0

        else:
            raise ValueError("Invalid boundary condition: {}".format(params['x1l_bound']))

        if params['coordinates'] != "minkowski" and params['x1l_inflow'] == 0:
            # Make sure there is no inflow at the inner boundary
            # inflow_check(G, P, left=True)
            pass

    if G.global_stop[0] == G.NTOT[1]:
        slc_to = (s.ghostr, s.b, s.b)
        if params["x1r_bound"] == "outflow":
            slc_from = (s.boundr_o, s.b, s.b)
            P[s.allv + slc_to] = P[s.allv + slc_from]
            if pflag is not None:
                pflag[slc_to] = pflag[slc_from]

            rescale = G.gdet[Loci.CENT.value, -NG, :] / G.gdet[Loci.CENT.value, s.ghostr, :]
            P[s.B1 + slc_to] *= rescale[:, :, None]
            P[s.B2 + slc_to] *= rescale[:, :, None]
            P[s.B3 + slc_to] *= rescale[:, :, None]

        elif params["x1r_bound"] == "periodic" and G.global_start[0] == 0:
            slc_from = (s.boundl, s.b, s.b)
            P[s.allv + slc_to] = P[s.allv + slc_from]
            if pflag is not None:
                pflag[slc_to] = pflag[slc_from]

        elif params['x1r_bound'] == "user":
            P[s.allv + slc_to] = user_bound_r(params, G, P, slc=slc_to)
            if pflag is not None:
                pflag[slc_to] = 0

        else:
            raise ValueError("Invalid boundary condition: {}".format(params['x1r_bound']))

        if params['coordinates'] != "minkowski" and params['x1r_inflow'] == 0:
            # Make sure there is no inflow at the inner boundary
            # inflow_check(G, P, left=False)
            pass

    # TODO RESTORE MPI
    # sync_mpi_bound_X1(P)

    ################################## X2 BOUNDARY ##########################################

    if G.global_start[1] == 0:
        if G.N[2] < G.NG:
            for j in range(G.GN[2]):
                P[:, s.a, j, s.b] = P[:, s.a, NG, s.b]
                if pflag is not None:
                    pflag[s.a, j, s.b] = pflag[s.a, NG, s.b]

        elif params['x2l_bound'] == "polar":
            slc_to = (s.a, s.ghostl, s.b)
            slc_from = (s.a, s.boundl_r, s.b)
            P[s.allv + slc_to] = P[s.allv + slc_from]
            if pflag is not None:
                pflag[slc_to] = pflag[slc_from]

            # Turn U2,B2 opposite (TODO should this be done?)
            P[s.U2 + slc_to] *= -1.
            P[s.B2 + slc_to] *= -1.

        elif params["x2l_bound"] == "periodic" and G.global_stop[1] == G.NTOT[2]:
            slc_to = (s.a, s.ghostl, s.b)
            slc_from = (s.a, s.boundr, s.b)
            P[s.allv + slc_to] = P[s.allv + slc_from]
            if pflag is not None:
                pflag[slc_to] = pflag[slc_from]

        else:
            raise ValueError("Invalid boundary condition: {}".format(params['x2l_bound']))

    if G.global_stop[1] == G.NTOT[2]:
        if params['x2r_bound'] == "polar":
            slc_to = (s.a, s.ghostr, s.b)
            slc_from = (s.a, s.boundr_r, s.b)
            P[s.allv + slc_to] = P[s.allv + slc_from]
            if pflag is not None:
                pflag[slc_to] = pflag[slc_from]

            # Turn U2,B2 opposite (TODO should this be done?)
            P[s.U2 + slc_to] *= -1.
            P[s.B2 + slc_to] *= -1.

        elif params["x2r_bound"] == "periodic" and G.global_start[1] == 0:
            slc_to = (s.a, s.ghostr, s.b)
            slc_from = (s.a, s.boundl, s.b)
            P[s.allv + slc_to] = P[s.allv + slc_from]
            if pflag is not None:
                pflag[slc_to] = pflag[slc_from]

        else:
            raise ValueError("Invalid boundary condition: {}".format(params['x2r_bound']))

    # sync_mpi_bound_X2(S)

    ################################## X3 BOUNDARY ##########################################

    if G.global_start[2] == 0:
        if G.N[3] < G.NG:
            for k in range(G.GN[3]):
                P[:, s.a, s.a, k] = P[:, s.a, s.a, NG]
                if pflag is not None:
                    pflag[s.a, s.a, k] = pflag[s.a, s.a, NG]

        elif params["x3l_bound"] == "periodic" and G.global_stop[2] == G.NTOT[3]:
            slc_to = (s.a, s.a, s.ghostl)
            slc_from = (s.a, s.a, s.boundr)
            P[s.allv + slc_to] = P[s.allv + slc_from]
            if pflag is not None:
                pflag[slc_to] = pflag[slc_from]

        else:
            raise ValueError("Invalid boundary condition: {}".format(params['x3l_bound']))

    if G.global_stop[2] == G.NTOT[3]:
        if params["x3r_bound"] == "periodic" and G.global_start[2] == 0:
            slc_to = (s.a, s.a, s.ghostr)
            slc_from = (s.a, s.a, s.boundl)
            P[s.allv + slc_to] = P[s.allv + slc_from]
            if pflag is not None:
                pflag[slc_to] = pflag[slc_from]

        else:
            raise ValueError("Invalid boundary condition: {}".format(params['x3r_bound']))

    # sync_mpi_bound_X3(S)


def inflow_check(params, G, P, left):
    s = G.slices
    sh = G.shapes
    ucon = ucon_calc(params, G, P, Loci.CENT)

    if left:
        slc = ucon[1] < 0
    else:
        slc = ucon[1] > 0.

    # Find gamma and remove it from prims near the edge
    gamma = mhd_gamma_calc(params['queue'], G, P, Loci.CENT)
    P[s.U1] /= gamma
    P[s.U2] /= gamma
    P[s.U3] /= gamma
    alpha = G.lapse[Loci.CENT.value]
    beta1 = G.gcon[Loci.CENT.value, 0, 1] * alpha**2

    # Reset radial velocity so radial 4-velocity is zero
    P[s.U1] = beta1/alpha

    # Now find new gamma and put it back in
    vsq = np.zeros_like(sh.grid_scalar)
    for mu in range(1, 4):
      for nu in range(1, 4):
        vsq += G.gcov[Loci.CENT.value, mu, nu, :, :, None] * \
               P[s.U1.value+mu-1, :, :, :] * \
               P[s.U1.value+nu-1, :, :, :]


    vsq[np.abs(vsq) < 1.e-13] = 1.e-13
    vsq[vsq >= 1.] = 1. - 1./(params['gamma_max']**2)

    gamma = 1./np.sqrt(1. - vsq)
    P[s.U1] *= gamma
    P[s.U2] *= gamma
    P[s.U3] *= gamma


def fix_flux(params, G, F):
    s = G.slices
    if G.global_start[0] == 0 and params['x1l_inflow'] == 0:
        F[1][s.RHO, G.NG, :, :] = np.clip(F[1][s.RHO, G.NG, :, :], None, 0.)

    if G.global_stop[0] == G.NTOT[1] and params['x1r_inflow'] == 0:
        F[1][s.RHO, G.N[1] + G.NG, :, :] = np.clip(F[1][s.RHO, G.N[1] + G.NG, :, :], 0., None)

    if G.global_start[1] == 0:
        F[1][s.B2, :, -1 + G.NG, :] = -F[1][s.B2, :, G.NG, :]
        F[3][s.B2, :, -1 + G.NG, :] = -F[3][s.B2, :, G.NG, :]
        F[2][:, :, G.NG, :] = 0.

    if G.global_stop[1] == G.NTOT[2]:
        F[1][s.B2, :, G.N[2] + G.NG, :] = -F[1][s.B2, :, G.N2 - 1 + G.NG, :]
        F[3][s.B2, :, G.N[2] + G.NG, :] = -F[3][s.B2, :, G.N2 - 1 + G.NG, :]
        F[2][:, :, G.N[2] + G.NG, :] = 0.

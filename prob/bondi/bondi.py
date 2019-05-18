# Bondi inflow problem: steady-state & symmetric solution

import numpy as np

from bounds import set_bounds, register_user_bound
from fixup import fixup
from coordinates import BL
from defs import Loci

from debug_tools import plot, plot_all_prims


# Adapted from M. Chandra
def get_Tfunc(T, r, n, C1, C2):
    return (1. + (1. + n) * T)**2. * (1. - 2. / r + (C1 / r**2 / (T**n))**2.) - C2


def get_T(r, n, C1, C2):
    # TODO parameter these?
    rtol = 1.e-10
    ftol = 1.e-14
    Tmin = 0.6 * (np.sqrt(C2) - 1.) / (n + 1)
    Tmax = (C1 * np.sqrt(2. / r**3))**(1./n)

    T0 = 0.6*Tmin*np.ones_like(r)
    f0 = get_Tfunc(T0, r, n, C1, C2)
    T1 = Tmax
    f1 = get_Tfunc(T1, r, n, C1, C2)

    if np.any(f0 * f1 > 0.):
        print("Failed solving for T at r = {} C1 = {} C2 = {}".format(r, C1, C2))
        exit(1)

    Th = (f1*T0 - f0*T1)/(f1 - f0)
    fh = get_Tfunc(Th, r, n, C1, C2)
    epsT = rtol*(Tmin + Tmax)
    iter_flag = np.logical_and(np.abs(Th - T0) > epsT, np.abs(Th - T1) > epsT, np.abs(fh) > ftol)
    while np.any(iter_flag):
        cond = (fh * f0) < 0
        slc0 = np.logical_and(cond, iter_flag)
        T0[slc0] = Th[slc0]
        f0[slc0] = fh[slc0]
        slc1 = np.logical_and(np.logical_not(cond), iter_flag)
        T1[slc1] = Th[slc1]
        f1[slc1] = fh[slc1]

        Th[iter_flag] = (f1[iter_flag] * T0[iter_flag] - f0[iter_flag] * T1[iter_flag])/(f1[iter_flag] - f0[iter_flag])
        fh[iter_flag] = get_Tfunc(Th[iter_flag], r[iter_flag], n, C1, C2)
        iter_flag = np.logical_and(np.abs(Th - T0) > epsT, np.abs(Th - T1) > epsT, np.abs(fh) > ftol)

    return Th


def fourvel_to_prim(ucon, P, G, slc=None):
    s = G.slices
    if slc is None:
        slc = s.all

    alpha = 1.0 / np.sqrt(-G.gcon[Loci.CENT.value, 0, 0])
    # We don't use the zero
    beta = alpha**2 * G.gcon[Loci.CENT.value, 0, :]
    gamma = ucon[0] * alpha[:, :, None]

    P[s.U1][slc] = (ucon[1] + beta[1, :, :, None]*gamma/alpha[:, :, None])[slc]
    P[s.U2][slc] = (ucon[2] + beta[2, :, :, None]*gamma/alpha[:, :, None])[slc]
    P[s.U3][slc] = (ucon[3] + beta[3, :, :, None]*gamma/alpha[:, :, None])[slc]

    return P


def set_ut(ucon, gcov):
    a = gcov[0, 0][:, :, None]*np.ones_like(ucon[1])
    b = 2.*(gcov[0, 1][:, :, None]*ucon[1] +
           gcov[0, 2][:, :, None]*ucon[2] +
           gcov[0, 3][:, :, None]*ucon[3])
    c = 1. + (gcov[1, 1][:, :, None]*ucon[1]**2 +
               gcov[2, 2][:, :, None]*ucon[2]**2 +
               gcov[3, 3][:, :, None]*ucon[3]**2) + \
         2. * (gcov[1, 2][:, :, None]*ucon[1]*ucon[2] +
               gcov[1, 3][:, :, None]*ucon[1]*ucon[3] +
               gcov[2, 3][:, :, None]*ucon[2]*ucon[3])

    discr = b**2 - 4.*a*c

    ucon[0] = (-b - np.sqrt(discr)) / (2. * a)

    return ucon


def get_prim_bondi(params, G, P, slc=None):
    s = G.slices

    if slc is None:
        slc = s.all

    n = 1./(params['gam'] - 1.)
    rs = params['rs']

    # Solution constants
    uc = np.sqrt(params['mdot'] / (2. * rs))
    Vc = -np.sqrt(uc**2 / (1. - 3. * uc**2))
    Tc = -n * Vc**2 / ((n + 1.) * (n * Vc**2 - 1.))
    C1 = uc * rs**2 * Tc**n
    C2 = (1. + (1. + n) * Tc)**2 * (1. - 2. * params['mdot'] / rs + C1**2 / (rs**4 * Tc**(2*n)))

    #print("uc, Vc, Tc, C1, C2: ", uc, Vc, Tc, C1, C2)

    X = G.coord_all()
    r, th, phi = G.coords.ks_coord(X)
    # COORDS: Ensure everything inside the horizon uses r_hor, to avoid singularity problems w/BL
    r = np.clip(r, G.coords.r_hor+0.01, None)
    BLC = BL(params)
    xBL = np.array([np.zeros_like(r)[:, :, G.NG], r[:, :, G.NG], th[:, :, G.NG], phi[:, :, G.NG]])

    T = get_T(r, n, C1, C2)
    ur = -C1/(T**n * r**2)
    rho = T**n
    u = rho * T / (params['gam'] - 1.)

    ucon_bl = np.zeros([4, *r.shape])
    ucon_bl[1] = ur

    ucon_bl = set_ut(ucon_bl, BLC.gcov(xBL))

    ucon_ks = np.einsum("ij...,j...->i...", BLC.dxdX(xBL)[:, :, :, :, None], ucon_bl)

    dxdX = G.coords.dxdX(X[:, :, G.NG]).transpose((2, 3, 0, 1))
    dXdx = np.linalg.inv(dxdX).transpose((2, 3, 0, 1))
    ucon_mks = np.einsum("ij...,j...->i...", dXdx, ucon_ks)

    P = fourvel_to_prim(ucon_mks, P, G, slc=slc)

    # TODO operate only on slices above to make boundaries go faster?
    P[s.RHO][slc] = rho[slc]
    P[s.UU][slc] = u[slc]
    P[s.B3VEC + slc] = 0.

    # Electrons make no physical sense here but are a very useful debug tool
    # At least set them consistently here to test deterministic evolution
    if params['electrons']:
        # Set electron internal energy to constant fraction of fluid internal energy
        uel = params['fel0']*P[s.UU]
        # Initialize entropies
        P[s.KTOT][slc] = ((params['gam'] - 1.) * P[s.UU] * P[s.RHO]**(-params['gam']))[slc]
        P[s.KEL][slc] =  ((params['gam_e'] - 1.) * uel * P[s.RHO]**(-params['gam_e']))[slc]


    return P[s.allv + slc]


def init(params, G, P):
    get_prim_bondi(params, G, P)

    # TODO handle logging better than print statements...
    print("a = {} Rhor = {}".format(G.coords.a, G.coords.r_hor))

    print("mdot = {}".format(params['mdot']))
    print("rs   = {}".format(params['rs']))
    # print("n    = {}", n)
    # print("C1   = {}", C1)
    # print("C2   = {}", C2)

    if params['electrons']:
        #init_electrons(G, S)
        pass

    # Register analytic condition as user right boundary
    register_user_bound(func_right=get_prim_bondi)

    # Enforce boundary conditions
    # TODO fixup takes cl_arrays...
    #fixup(params, G, P)
    set_bounds(params, G, P)

    if 'plot' in params and params['plot']:
        plot_all_prims(G, P, "Pstart")


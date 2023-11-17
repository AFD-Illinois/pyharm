__license__ = """
 File: fm_torus.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020-2023, Ben Prather and AFD Group at UIUC
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from scipy.optimize import brentq
import warnings

from pyharm.coordinates import BL
from pyharm.fluid_state import FluidState

from pyharm.grmhd.init_tools import set_fourvel_t, fourvel_to_prim

__doc__ = \
"""Compute the Fishbone-Moncrief torus solution.
Most code originally from Patrick Mullen & Chris White,
adapted for pyharm.
HARM versions are from iharm primordium.
"""

## FISHBONE-MONCRIEF
def get_fm_torus_fluid_state(G, r_in=6.0, r_max=12.0, gamma=5./3, rho_max=1.0,
                             add_atmo=True, rho_atmo=1e-6, u_atmo=1e-8,
                             use_harm_functions=True, kappa=1e-3):
    """Calculate FM torus"""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Preliminary
        # Make sure output is full grid size, even though we're symmetric in X3
        a, n1, n2, n3 = G['a'], G['n1'], G['n2'], G['n3']
        ones = np.ones((n1, n2, n3))

        # Density & energy
        if use_harm_functions:
            l =_lfish_calc(a, r_max)
            lnh = _lnh_calc(a, l, r_in, G['r']*ones, G['th']*ones)
            hm1   = np.exp(lnh) - 1.
            rho = (hm1 * (gamma - 1.) / (kappa * gamma))**(1. / (gamma - 1.))
            u   = kappa * (rho**gamma) / (gamma - 1.)
        else:
            f = _fm_f(G['r']*ones, G['th']*ones, r_max, a)
            f_in = _fm_f(r_in, np.pi/2, r_max, a)
            f_peak = _fm_f(r_max, np.pi/2, r_max, a)
            h = _fm_h(f, f_in)
            h_peak = _fm_h(f_peak, f_in)
            rho = _calculate_rho(h, h_peak, rho_max, gamma)
            u = _calculate_u(h, gamma, rho)
            # Erase inner portions
            rho = np.where(G['r'] >= r_in, rho, np.nan)
            u = np.where(G['r'] >= r_in, u, np.nan)

        if add_atmo:
            # Atmosphere to replace all NaN
            floor_rho = rho_atmo*G['r']**(-3)*ones
            floor_u = u_atmo*G['r']**(-3)*ones
            rho = np.where(np.isnan(rho), floor_rho, rho)
            u = np.where(np.isnan(u), floor_u, u)

        cache = {}
        cache['RHO'] = cache['rho'] = rho
        cache['UU'] = cache['u'] = u

        # Velocity (always HARM function)
        l =_lfish_calc(a, r_max)
        utilde = _utilde_calc(G, G['r']*ones, G['th']*ones, l, a)
        cache['U1'] = utilde[0]
        cache['U2'] = utilde[1]
        cache['U3'] = utilde[2]

        # TODO tilt?

        # Add the parameters
        params = {}
        params['r_in'] = r_in
        params['r_max'] = r_max
        params['rho_max'] = rho_max
        params['gam'] = gamma

    return FluidState(cache, params=params, grid=G)

# HARM versions of supporting functions
def _lfish_calc(a, r):
    """Calculate disk angular momentum from r_in, r_max"""
    sqtr = np.sqrt(r)
    return ((a*a - 2. * a * sqtr + r*r) *
             ((-2. * a * r * (a*a - 2. * a * sqtr + r*r)) /
                  np.sqrt(2. * a * sqtr + (-3. + r) * r) +
              ((a + (-2. + r) * sqtr) * (r*r*r + a*a * (2. + r))) /
                  np.sqrt(1 + (2. * a) / r**1.5 - 3. / r))) / \
            (r*r*r * np.sqrt(2. * a * sqtr + (-3. + r) * r) *
             (a*a + (-2. + r) * r))

def _lnh_calc(a, l, r_in, r, th):
    """Fishbone-Moncrief log-enthalpy parameter"""
    # TODO split, metric quantities more clearly
    sth = np.sin(th)
    cth = np.cos(th)

    r2 = r*r
    a2 = a*a
    DD = r2 - 2. * r + a2
    AA = (r2 + a2)**2 - DD * a2 * sth * sth
    SS = r2 + a2 * cth * cth

    thin = np.pi / 2.
    sthin = np.sin(thin)
    cthin = np.cos(thin)

    rin2 = r_in**2
    DDin = rin2 - 2. * r_in + a2
    AAin = (rin2 + a2)**2 - DDin * a2 * sthin * sthin
    SSin = rin2 + a2 * cthin * cthin

    rho =   0.5 * \
                np.log((1. +
                        np.sqrt(1. +
                            4. * (l * l * SS * SS) * DD / (AA * AA * sth * sth))) /
                    (SS * DD / AA)) - \
            0.5 * np.sqrt(1. +
                        4. * (l * l * SS * SS) * DD /
                            (AA * AA * sth * sth)) - \
            2. * a * r * l / AA - \
            (0.5 *
                    np.log((1. +
                        np.sqrt(1. +
                            4. * (l * l * SSin * SSin) * DDin /
                                (AAin * AAin * sthin * sthin))) /
                        (SSin * DDin / AAin)) -
                0.5 * np.sqrt(1. +
                        4. * (l * l * SSin * SSin) * DDin / (AAin * AAin * sthin * sthin)) -
                2. * a * r_in * l / AAin)

    return np.where(r > r_in, rho, np.nan)

def _utilde_calc(G, r, th, l, a):
    # Coordinate stuff we'll need
    bl = BL({'a': a})
    sth = np.sin(th)
    X_bl = G.coords.ks_coord(G['X'], fourv=True)
    delta = bl.delta(X_bl)
    sigma = bl.sigma(X_bl)
    aa    = bl.aa(X_bl)

    expm2chi = sigma * sigma * delta / (aa * aa * sth * sth)
    up1      = np.sqrt((-1. + np.sqrt(1. + 4. * l * l * expm2chi)) / 2.)
    ucon3       = 2. * a * G['r'] * np.sqrt(1. + up1 * up1) / np.sqrt(aa * sigma * delta) + \
                       np.sqrt(sigma / aa) * up1 / sth
    
    ucon_bl = np.zeros([4, *r.shape])

    # We have ucon & must convert to primitive U1,2,3 in native coords
    set_fourvel_t(G['gcov_bl'], ucon_bl)
    # Convert ucon(Bl) to ucon(KS)
    ucon_ks = G.bl_to_ks_con(ucon_bl)
    # Convert ucon(KS) to ucon(MKS/FMKS)
    ucon_mks = G.ks_to_native_con(ucon_ks)
    # Convert to primitive vars
    return fourvel_to_prim(G['gcon'], ucon_mks)

# Athena/cwhite versions
def _fm_ls(a, r_max):
    """Calculate disk angular momentum from r_in, r_max"""
    return ((1.0 / r_max**3) ** 0.5*((r_max**4 + a**2 * r_max**2 - 2.0*a**2 * r_max
                                       - a * (r_max)**0.5 * (r_max**2 - a**2)) /
                                      (r_max**2 - 3.0 * r_max + 2.0*a * (r_max)**0.5)))

def _fm_f(r, th, r_max, a):
    """Fishbone-Moncrief log-enthalpy parameter"""
    bl = BL({'a': a})
    s = np.sin(th)
    x = np.array([np.zeros_like(r),r,th,np.zeros_like(r)])
    delta = bl.delta(x)
    sigma = bl.sigma(x)
    aa    = bl.aa(x)
    ls = _fm_ls(a, r_max)
    f = (0.5 * np.log(aa/(delta*sigma) + ((aa/(delta*sigma))**2
                        + 4.0*ls**2/(delta*s**2)) ** 0.5)
            - 0.5 * (1.0 + 4.0*ls**2*delta*sigma**2/(aa**2 * s**2)) ** 0.5 - 2.0*a*r*ls/aa)
    return f

def _fm_h(f, f_in):
    """Fishbone-Moncrief enthalpy"""
    h = np.exp(f - f_in)
    h = np.where(h > 1.0, h, np.nan)
    return h

## CHAKRABARTI
def get_c_torus_fluid_state(G, r_in=6.0, r_max=12.0, gamma=5./3, rho_max=1.0):
    """Return a FluidState of the Chakrabarti torus. Density only!"""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Ensure proper array sizes
        ones = np.ones((G['n1'], G['n2'], G['n3']))
        r = G['r']*ones
        th = G['th']*ones
        a = G['a']

        c, n = _c_cn(r_in, r_max, a)
        ll = _c_l(r, th, c, n, a)
        l_peak = _c_l(r_max, np.pi/2.0, c, n, a)
        h = _c_h(r, th, ll, c, n, r_in, a)
        h_peak = _c_h(r_max, np.pi/2.0, l_peak, c, n, r_in, a)
        rho = _calculate_rho(h, h_peak, rho_max, gamma)
        u = _calculate_u(h, gamma, rho)
        cache = {}
        cache['RHO'] = cache['rho'] = np.where(G['r'] >= r_in, rho, np.nan)
        cache['UU'] = cache['u'] = np.where(G['r'] >= r_in, u, np.nan)

        # Add the parameters
        params = {}
        params['r_in'] = r_in
        params['r_max'] = r_max
        params['rho_max'] = rho_max
        params['gam'] = gamma

    return FluidState(cache, params=params, grid=G)

def _c_cn(r_in, r_max, a):
    """Chakrabarti c and n parameters"""
    l_in = _l_k(r_in, a)
    l_peak = _l_k(r_max, a)
    lambda_in = _vz(r_in, l_in, a)
    lambda_peak = _vz(r_max, l_peak, a)
    n = np.log(l_peak/l_in) / np.log(lambda_peak/lambda_in)
    c = l_in*lambda_in**(-n)
    return c, n

def _c_l(r, th, c, n, a):
    """Chakrabarti ll parameter"""
    bl = BL({'a': a})
    x = np.array([np.zeros_like(r),r,th,np.zeros_like(r)])
    gcov = bl.gcov(x)
    g_tt, g_tphi, g_phiphi = gcov[0,0], gcov[0,3], gcov[3,3]

    l_val = np.zeros_like(r)
    if np.isscalar(l_val) or len(l_val.shape) == 0:
        res = lambda ll : (ll/c)**(2.0/n) + (ll*g_phiphi + ll**2*g_tphi) / \
                            (g_tphi + ll*g_tt)
        try:
            l_val = brentq(res, 1.0, 100.0)
        except ValueError:
            l_val = np.nan
    else:
        res = lambda ll,i,j : (ll/c)**(2.0/n) + (ll*g_phiphi[i,j,0] + ll**2*g_tphi[i,j,0]) / \
                            (g_tphi[i,j,0] + ll*g_tt[i,j,0])
        # Only do the optimization in 2D, it's expensive
        for i in range(l_val.shape[0]):
            for j in range(l_val.shape[1]):
                try:
                    l_val[i,j,:] = brentq(res, 1.0, 100.0, args=(i,j))
                except ValueError:
                    l_val[i,j,:] = np.nan

    return l_val

def _c_h(r, theta, ll, c, n, r_in, a):
    """Chakrabarti enthalpy"""
    l_in = _c_l(r_in, np.pi/2.0, c, n, a)
    u_t = _c_u_t(r, theta, ll, a)
    u_t_in = _c_u_t(r_in, np.pi/2.0, l_in, a)
    h = u_t_in / u_t
    if n == 1.0:
        h *= (l_in/ll)**(c**2/(c**2-1.0))
    else:
        h *= (abs(1.0 - c**(2.0/n)*ll**(2.0-2.0/n))**(n/(2.0-2.0*n)) *
              abs(1.0 - c**(2.0/n)*l_in**(2.0-2.0/n))**(n/(2.0*n-2.0)))
    return h

def _c_u_t(r, th, ll, a):
    """Chakrabarti u_t"""
    bl = BL({'a': a})
    x = np.array([np.zeros_like(r),r,th,np.zeros_like(r)])
    gcov = bl.gcov(x)
    g_tt, g_tphi, g_phiphi = gcov[0,0], gcov[0,3], gcov[3,3]
    return -((g_tphi**2 - g_tt*g_phiphi)/(g_phiphi + 2.0*ll*g_tphi + ll**2*g_tt))**0.5

## GENERAL
def _calculate_rho(h, h_peak, rho_max, gamma):
    """Density from enthalpy"""
    tt = (gamma-1.0)/gamma*(h - 1.0)
    tt_peak = (gamma-1.0)/gamma*(h_peak - 1.0)
    return rho_max*(tt / tt_peak)**(1.0/(gamma-1.0))

def _calculate_u(h, gamma, rho):
    """Internal energy from enthalpy & density"""
    return (h - 1.0)/gamma*rho

def _l_k(r, a):
    """Keplerian ll"""
    return (r**0.5*((1.0 - 2.0*a*(1.0/r)**0.5/r + a**2/r**2) /
                    (1.0 - 2.0/r + a*(1.0/r)**0.5/r)))

def _vz(r, ll, a):
    """Von Zeipel parameter"""
    return np.sqrt((r**3 + a**2*r + 2.0*a*(a-ll)) / (r + 2*a/ll - 2.0))
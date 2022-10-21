__license__ = """
 File: variables.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020-2022, AFD Group at UIUC
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

__doc__ = \
"""Functions for calculating various quantities in terms of the primitive variables and geometry.
"""

import numpy as np

from .defs import Loci
from .grmhd.b_field import *

# Define a dict of names, coupled with the functions required to obtain their variables.
# That way, we only need to specify lists and final operations in eht_analysis,
# AND don't need to cart all these things around in memory
fns_dict = {# 4-vectors
            'ucon': lambda dump: ucon_calc(dump),
            'ucov': lambda dump: dump.grid.lower_grid(dump['ucon']),
            'bcon': lambda dump: bcon_calc(dump),
            'bcov': lambda dump: dump.grid.lower_grid(dump['bcon']),
            # Versions in base coordinates
            # these use the reverse of dxdX/dXdx as they transform *back*
            'ucon_base': lambda dump: np.einsum("i...,ij...->j...", dump["ucon"], dump['dxdX']),
            'ucov_base': lambda dump: np.einsum("i...,ij...->j...", dump["ucov"], dump['dXdx']),
            'bcon_base': lambda dump: np.einsum("i...,ij...->j...", dump["bcon"], dump['dxdX']),
            'bcov_base': lambda dump: np.einsum("i...,ij...->j...", dump["bcov"], dump['dXdx']),
            # Versions in Cartesian
            'ucon_cart': lambda dump: np.einsum("i...,ij...->j...", dump["ucon_base"], dump['dXdx_cart']),
            'ucov_cart': lambda dump: np.einsum("i...,ij...->j...", dump["ucov_base"], dump['dxdX_cart']),
            'bcon_cart': lambda dump: np.einsum("i...,ij...->j...", dump["bcon_base"], dump['dXdx_cart']),
            'bcov_cart': lambda dump: np.einsum("i...,ij...->j...", dump["bcov_base"], dump['dxdX_cart']),
            # Miscallany!
            'bsq': lambda dump: dump.grid.dot(dump['bcov'], dump['bcon']),
            'sigma': lambda dump: dump['bsq'] / dump['RHO'],
            'u': lambda dump: dump['UU'],
            'h': lambda dump: enthalpy(dump),
            'FM': lambda dump: dump['RHO'] * dump['ucon'][1],
            'FE': lambda dump: -T_mixed(dump, 1, 0),
            'FE_EM': lambda dump: -TEM_mixed(dump, 1, 0),
            'FE_Fl': lambda dump: -TFl_mixed(dump, 1, 0),
            'FE_PAKE': lambda dump: -TPAKE_mixed(dump, 1, 0),
            'FE_EN': lambda dump: -TEN_mixed(dump, 1, 0),
            'FE_norho': lambda dump: -T_mixed(dump, 1, 0) - dump['rho']*dump['ucon'][1],
            'FL': lambda dump: T_mixed(dump, 1, 3),
            'FL_EM': lambda dump: TEM_mixed(dump, 1, 3),
            'FL_Fl': lambda dump: TFl_mixed(dump, 1, 3),
            'Be_b': lambda dump: bernoulli(dump, with_B=True),
            'Be_nob': lambda dump: bernoulli(dump, with_B=False),
            'Pg': lambda dump: (dump['gam'] - 1.) * dump['UU'],
            'p': lambda dump: dump['Pg'],
            'Pb': lambda dump: dump['bsq'] / 2,
            'Ptot': lambda dump: dump['Pg'] + dump['Pb'],
            'beta': lambda dump: dump['Pg'] / dump['Pb'],
            'jcov': lambda dump: dump.grid.lower_grid(dump['jcon']),
            'jsq': lambda dump: dump.grid.dot(dump['jcon'], dump['jcov']),
            'Jsq': lambda dump: dump.grid.dot(dump['jcon'], dump['jcov']) + dump.grid.dot(dump['jcon'], dump['ucov'])**2,
            'b': lambda dump: np.sqrt(dump['bsq']),
            'Gamma': lambda dump: lorentz_calc(dump),
            'betagamma': lambda dump: np.sqrt((dump['FE'] / dump['FM'])**2 - 1),
            'Theta': lambda dump: (dump['gam'] - 1) * dump['UU'] / dump['RHO'],
            'Thetap': lambda dump: (dump['gam_p'] - 1) * dump['UU'] / dump['RHO'],
            'Thetae': lambda dump: (dump['gam_e'] - 1) * dump['UU'] / dump['RHO'],
            'Thetae_rhigh': lambda dump: thetae_rhigh(dump),
            'JE0': lambda dump: -T_mixed(dump, 0, 0),
            'JE1': lambda dump: -T_mixed(dump, 1, 0),
            'JE2': lambda dump: -T_mixed(dump, 2, 0),
            'lam_MRI': lambda dump: lam_MRI(dump),
            'lam_MRI_old': lambda dump: lam_MRI_old(dump),
            'lam_MRI_transform': lambda dump: lam_MRI_transform(dump),
            'vA': lambda dump: alfven_speed(dump),
            'jet_psi': lambda dump: jet_psi(dump),
            'divB': lambda dump: divB(dump.grid, dump['B']),
            'divB_cons': lambda dump: divB_cons(dump.grid, dump['cons.B']),
            'lumproxy': lambda dump: lum_proxy(dump),
            'jI': lambda dump: jnu(dump),
            'K': lambda dump: (dump['gam']-1.) * dump['UU'] * pow(dump['RHO'], -dump['gam']),
            }

## Physics functions ##

def lorentz_calc(dump, loc=Loci.CENT):
    """Find relativistic gamma-factor w.r.t. normal observer"""
    if 'ucon' in dump.cache:
        return dump['ucon'][0] * dump['lapse']
    else:
        G = dump.grid
        return np.sqrt(1 + (G.gcov[loc.value, 1, 1] * dump['U1'] ** 2 +
                            G.gcov[loc.value, 2, 2] * dump['U2'] ** 2 +
                            G.gcov[loc.value, 3, 3] * dump['U3'] ** 2) + \
                            2. * (G.gcov[loc.value, 1, 2] * dump['U1'] * dump['U2'] +
                                  G.gcov[loc.value, 1, 3] * dump['U1'] * dump['U3'] +
                                  G.gcov[loc.value, 2, 3] * dump['U2'] * dump['U3']))

def ucon_calc(dump, loc=Loci.CENT):
    """Find contravariant fluid four-velocity"""
    G = dump.grid
    ucon = np.zeros((4, *dump['U1'].shape))
    ucon[0] = dump['Gamma'] / dump['lapse']
    for mu in range(1, 4):
        ucon[mu] = dump['uvec'][mu-1] - dump['Gamma'] * dump['lapse'] * G.gcon[loc.value, 0, mu]

    return ucon


def bcon_calc(dump):
    """Calculate magnetic field four-vector"""
    bcon = np.zeros_like(dump['ucon'])
    bcon[0] = dump['B'][0] * dump['ucov'][1] + \
              dump['B'][1] * dump['ucov'][2] + \
              dump['B'][2] * dump['ucov'][3]
    for mu in range(1, 4):
        bcon[mu] = (dump['B'][mu-1] + bcon[0] * dump['ucon'][mu]) \
                        / dump['ucon'][0]

    return bcon

# These are separated because raising/lowering is slow
def T_con(dump, i, j):
    gam = dump['gam']
    return ((dump['RHO'] + gam * dump['UU'] + dump['bsq']) * dump['ucon'][i] * dump['ucon'][j] +
            ((gam - 1) * dump['UU'] + dump['bsq'] / 2) * dump['gcon'][i, j] - dump['bcon'][i] *
            dump['bcon'][j])


def T_cov(dump, i, j):
    gam = dump['gam']
    return ((dump['RHO'] + gam * dump['UU'] + dump['bsq']) * dump['ucov'][i] * dump['ucov'][j] +
            ((gam - 1) * dump['UU'] + dump['bsq'] / 2) * dump['gcov'][i, j] - dump['bcov'][i] *
            dump['bcov'][j])


def T_mixed(dump, i, j):
    gam = dump['gam']
    if i != j:
        return ((dump['RHO'] + gam * dump['UU'] + dump['bsq']) * dump['ucon'][i] * dump['ucov'][j] +
                - dump['bcon'][i] * dump['bcov'][j])
    else:
        return ((dump['RHO'] + gam * dump['UU'] + dump['bsq']) * dump['ucon'][i] * dump['ucov'][j] +
                (gam - 1) * dump['UU'] + dump['bsq'] / 2 - dump['bcon'][i] * dump['bcov'][j])


# "Sub-tensors" representing components of the energy, only used as mixed so far.
def TEM_mixed(dump, i, j):
    if i != j:
        return dump['bsq'] * dump['ucon'][i] * dump['ucov'][j] - \
               dump['bcon'][i] * dump['bcov'][j]
    else:
        return dump['bsq'] * dump['ucon'][i] * dump['ucov'][j] + dump['bsq'] / 2 - \
               dump['bcon'][i] * dump['bcov'][j]

def TPAKE_mixed(dump, i, j):
    if j != 0:
        return dump['RHO'] * dump['ucov'][j] * dump['ucon'][i]
    else:
        return dump['RHO'] * (dump['ucov'][j] + 1) * dump['ucon'][i]

def TEN_mixed(dump, i, j):
    gam = dump['gam']
    if i != j:
        # (u + p) u^i u_j + p delta(i,j)
        return (gam * dump['UU']) * dump['ucon'][i] * dump['ucov'][j]
    else:
        return (gam * dump['UU']) * dump['ucon'][i] * dump['ucov'][j] + (gam - 1) * dump['UU']

def TFl_mixed(dump, i, j):
    gam = dump['gam']
    if i != j:
        return (dump['RHO'] + gam * dump['UU']) * dump['ucon'][i] * dump['ucov'][j]
    else:
        return (dump['RHO'] + gam * dump['UU']) * dump['ucon'][i] * dump['ucov'][j] + (gam - 1) * dump['UU']


def F_con(dump, i, j):
    """Return the i,j component of contravariant Maxwell tensor"""

    Fconij = np.zeros_like(dump['RHO'])
    if i != j:
        for mu in range(4):
            for nu in range(4):
                Fconij += (- _antisym(i, j, mu, nu) / dump['gdet']) * dump['ucov'][mu] * dump['bcov'][nu]

    return Fconij


def F_cov(dump, i, j):
    """Return the i,j component of covariant Maxwell tensor"""
    Fcovij = np.zeros_like(dump['RHO'])
    for mu in range(4):
        for nu in range(4):
            Fcovij += F_con(dump, mu, nu) * dump['gcov'][mu, i] * dump['gcov'][nu, j]

    return Fcovij

# F_mixed?

def bernoulli(dump, with_B=False):
    if with_B:
        #return -(T_mixed(dump, 0, 0) / dump['FM']) - 1
        return np.sqrt( (-T_mixed(dump, 1, 0) / (dump['rho']*dump['u^1']))**2 - 1)
    else:
        return -(1 + dump['gam'] * dump['UU'] / dump['RHO']) * dump['ucov'][0] - 1

def lam_MRI_old(dump):
    return (2*np.pi)/(dump['u^3']/dump['u^0']) * dump['b^th']/np.sqrt(dump['rho'] + dump['u'] + dump['p'] + dump['bsq'])

def alfven_speed(dump):
    return dump['b']/np.sqrt(4*np.pi*dump['rho'])

def lam_MRI(dump):
    return dump['vA'] / (dump['u^3']/dump['u^0'])

def lam_MRI_transform(dump):
    # From Porth et al (2019) & referenced Takahashi 
    return 2 * np.pi / (np.sqrt(dump['rho']*dump['h'] + dump['bsq']) * (dump['u^3']/dump['u^0'])) * \
            dump['b^th'] * np.sqrt(dump['r']**2 + dump['a']**2 * np.cos(dump['th'])**2)

def enthalpy(dump):
    return 1 + dump['Pg'] + dump['u']

def jet_psi(dump):
    sig = dump['sigma']
    return np.where(sig >= 1, 1, np.where(sig <= 0.1, 0, sig))

def lum_proxy(dump):
    # See EHT code comparison paper
    return dump['rho'] ** 3 / dump['Pg'] ** 2 * np.exp(-0.2 * (dump['rho'] ** 2 / (dump['b'] * dump['Pg'] ** 2)) ** (1. / 3.))

def thetae_rhigh(dump, Rlow=1, Rhigh=10, beta_crit=1.0):
    units = dump.units
    betasq = dump['beta']**2 / beta_crit**2
    game = dump['gam_e']; gamp = dump['gam_p']
    trat = Rhigh * betasq/(1. + betasq) + Rlow /(1. + betasq)
    Thetae_unit = (units['MP']/units['ME']) * \
                  (game-1.) * (gamp-1.) / ( (gamp-1.) + (game-1.) * trat)
    return Thetae_unit * dump['Theta']

def jnu(dump, nu=230e9, theta=np.pi/3):
    units = dump.units
    Thetae = dump['Thetae_rhigh']
    K2 = 2 * Thetae**2 # Approximate Bessel K_2(1/Thetae)
    nuc = units['EE'] * dump['b'] / (2. * np.pi * units['ME'] * units['CL'])
    nus = (2. / 9.) * nuc * Thetae ** 2 * np.sin(theta)
    x = nu / nus
    f = (np.sqrt(x) + 2**(11/12) * x**(1/6))**2
    Ne = dump['RHO'] * units['RHO_unit'] / (units['MP'] + units['ME'])
    j = (np.sqrt(2) * np.pi * units['EE']**2 * Ne * nus / \
        (3. * units['CL'] * K2)) * f * np.exp(-x**(1/3))
    j[nu > 1.e12 * nus] = 0.
    return j / nu**2


## Internal functions ##

# Completely antisymmetric 4D symbol
# TODO cache? Is this validation necessary?
def _antisym(a, b, c, d):
    # Check for valid permutation
    if (a < 0 or a > 3): return 100
    if (b < 0 or b > 3): return 100
    if (c < 0 or c > 3): return 100
    if (d < 0 or d > 3): return 100

    # Entries different?
    if (a == b or a == c or a == d or
            b == c or b == d or c == d):
        return 0

    return _pp([a, b, c, d])


# Due to Norm Hardy; good for general n
def _pp(P):
    v = np.zeros_like(P)

    p = 0
    for j in range(len(P)):
        if v[j]:
            p += 1
        else:
            x = j
            while True:
                x = P[x]
                v[x] = 1
                if x == j:
                    break

    if p % 2 == 0:
        return 1
    else:
        return -1

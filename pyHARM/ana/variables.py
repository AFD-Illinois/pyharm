# Convenient analysis functions for physical calculations and averages
# Largely can be used via IharmDump objects, see iharm_dump.py

import os
import sys
import numpy as np

from pyHARM.diag import divB

# Define a dict of names, coupled with the functions required to obtain their variables.
# That way, we only need to specify lists and final operations in eht_analysis,
# AND don't need to cart all these things around in memory
fns_dict = {'rho': lambda dump: dump['RHO'],
            'bsq': lambda dump: dump.grid.dot(dump['bcov'], dump['bcon']),
            'sigma': lambda dump: dump['bsq'] / dump['RHO'],
            'U': lambda dump: dump['UU'],
            'u_t': lambda dump: dump['ucov'][0],
            'u^t': lambda dump: dump['ucon'][0],
            'u_r': lambda dump: dump['ucov'][1],
            'u^r': lambda dump: dump['ucon'][1],
            'u_phi': lambda dump: dump['ucov'][3],
            'u^phi': lambda dump: dump['ucon'][3],
            'FM': lambda dump: dump['RHO'] * dump['ucon'][1],
            'FE': lambda dump: -T_mixed(dump, 1, 0),
            'FE_EM': lambda dump: -TEM_mixed(dump, 1, 0),
            'FE_Fl': lambda dump: -TFl_mixed(dump, 1, 0),
            'FL': lambda dump: T_mixed(dump, 1, 3),
            'FL_EM': lambda dump: TEM_mixed(dump, 1, 3),
            'FL_Fl': lambda dump: TFl_mixed(dump, 1, 3),
            'Be_b': lambda dump: bernoulli(dump, with_B=True),
            'Be_nob': lambda dump: bernoulli(dump, with_B=False),
            'Pg': lambda dump: (dump.header['gam'] - 1.) * dump['UU'],
            'Pb': lambda dump: dump['bsq'] / 2,
            'Ptot': lambda dump: dump['Pg'] + dump['Pb'],
            'beta': lambda dump: dump['Pg'] / dump['Pb'],
            'betainv': lambda dump: 1 / dump['beta'],
            'jcov': lambda dump: dump.grid.lower_grid(dump['jcon']),
            'jsq': lambda dump: dump.grid.dot(dump['jcon'], dump['jcov']),
            'current': lambda dump: dump.grid.dot(dump['jcon'], dump['jcov']) + dump.grid.dot(dump['jcon'], dump['ucov'])**2,
            'B': lambda dump: np.sqrt(dump['bsq']),
            'betagamma': lambda dump: np.sqrt((dump['FE_EM'] + dump['FE_Fl']) / dump['FM'] - 1),
            'Theta': lambda dump: (dump.header['gam'] - 1) * dump['UU'] / dump['RHO'],
            'Thetap': lambda dump: (dump.header['gam_p'] - 1) * dump['UU'] / dump['RHO'],
            'Thetae': lambda dump: (dump.header['gam_e'] - 1) * dump['UU'] / dump['RHO'],
            'JE0': lambda dump: T_mixed(dump, 0, 0),
            'JE1': lambda dump: T_mixed(dump, 1, 0),
            'JE2': lambda dump: T_mixed(dump, 2, 0),
            'divB': lambda dump: divB(dump.grid, dump.prims)
            }


## Physics functions ##

# These are separated because raising/lowering is slow
def T_con(dump, i, j):
    gam = dump.header['gam']
    return ((dump['RHO'] + gam * dump['UU'] + dump['bsq']) * dump['ucon'][i] * dump['ucon'][j] +
            ((gam - 1) * dump['UU'] + dump['bsq'] / 2) * dump['gcon'][i, j, :, :, None] - dump['bcon'][i] *
            dump['bcon'][j])


def T_cov(dump, i, j):
    gam = dump.header['gam']
    return ((dump['RHO'] + gam * dump['UU'] + dump['bsq']) * dump['ucov'][i] * dump['ucov'][j] +
            ((gam - 1) * dump['UU'] + dump['bsq'] / 2) * dump['gcov'][i, j, :, :, None] - dump['bcov'][i] *
            dump['bcov'][j])


def T_mixed(dump, i, j):
    gam = dump.header['gam']
    if i != j:
        return ((dump['RHO'] + gam * dump['UU'] + dump['bsq']) * dump['ucon'][i] * dump['ucov'][j] +
                - dump['bcon'][i] * dump['bcov'][j])
    else:
        return ((dump['RHO'] + gam * dump['UU'] + dump['bsq']) * dump['ucon'][i] * dump['ucov'][j] +
                (gam - 1) * dump['UU'] + dump['bsq'] / 2 - dump['bcon'][i] * dump['bcov'][j])


def TEM_mixed(dump, i, j):
    if i != j:
        return dump['bsq'] * dump['ucon'][i] * dump['ucov'][j] - \
               dump['bcon'][i] * dump['bcov'][j]
    else:
        return dump['bsq'] * dump['ucon'][i] * dump['ucov'][j] + dump['bsq'] / 2 - \
               dump['bcon'][i] * dump['bcov'][j]


def TFl_mixed(dump, i, j):
    gam = dump.header['gam']
    if i != j:
        return (dump['RHO'] + gam * dump['UU']) * dump['ucon'][i] * dump['ucov'][j]
    else:
        return (dump['RHO'] + gam * dump['UU']) * dump['ucon'][i] * dump['ucov'][j] + (gam - 1) * dump['UU']


def Fcon(dump, i, j):
    """Return the i,j component of contravariant Maxwell tensor"""
    # TODO loopy this for currents on the backend & use results here

    Fconij = np.zeros_like(dump['RHO'])
    if i != j:
        for mu in range(4):
            for nu in range(4):
                Fconij[:, :, :] += _antisym(i, j, mu, nu) * dump['ucov'][mu] * dump['bcov'][nu]

    # Remember we want gdet in the vectors' coordinate system (this matters for KORAL dump files)
    # TODO is normalization correct?
    return Fconij * dump['gdet'][:, :, None]


def Fcov(dump, i, j):
    """Return the i,j component of covariant Maxwell tensor"""
    Fcovij = np.zeros_like(dump['RHO'])
    for mu in range(4):
        for nu in range(4):
            Fcovij += Fcon(dump, mu, nu) * dump['gcov'][mu, i, :, :, None] * dump['gcov'][nu, j, :, :, None]

    return Fcovij


def bernoulli(dump, with_B=False):
    if with_B:
        return -T_mixed(dump, 0, 0) / (dump['RHO'] * dump['ucon'][0]) - 1
    else:
        return -(1 + dump.header['gam'] * dump['UU'] / dump['RHO']) * dump['ucov'][0] - 1


# TODO needs work...
def jnu_inv(nu, Thetae, Ne, B, theta):
    K2 = 2. * Thetae ** 2
    nuc = EE * B / (2. * np.pi * ME * CL)
    nus = (2. / 9.) * nuc * Thetae ** 2 * np.sin(theta)
    j[nu > 1.e12 * nus] = 0.
    x = nu / nus
    f = pow(pow(x, 1. / 2.) + pow(2., 11. / 12.) * pow(x, 1. / 6.), 2)
    j = (np.sqrt(2.) * np.pi * EE ** 2 * Ne * nus / (3. * CL * K2)) * f * exp(-pow(x, 1. / 3.))
    return j / nu ** 2


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

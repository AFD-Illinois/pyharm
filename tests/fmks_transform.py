#!/usr/bin/env python3

# Test our geometry functions make sense.
# Example numbers taken from Illinois docs wiki:
# https://github.com/AFD-Illinois/docs/wiki/fmks-and-you

import numpy as np
from pyharm.grid import Grid

def compare(a, b):
    if not np.all(np.logical_or(np.logical_and(a < 1e-16, b < 1e-16), (b - a)/b < 1e-8)):
        print("A:", a)
        print("B:", b)
        print("Absolute: ", b - a)
        print("Relative: ", (b - a)/b)
        return False
    else:
        return True

# Values in FMKS from a MAD simulation
# zone 11,12,13 of dump 1200 of MAD a+0.9375 384x192x192 iharm3D eht_v1 run
P = np.array([0, 0, 0.4553683,  0.0147898,  0.7197036, 3.6612415,  0.2197483, -5.5480947])

# Metric parameters from that simulation
params = {'coordinates': 'fmks', 'a': 0.9375,
          'r_in': 1.2175642950007606, 'r_out': 1000.0,
          'hslope': 0.3, 'mks_smooth': 0.5, 'poly_xt': 0.82, 'poly_alpha': 14.0,
          'n1': 384, 'n2': 192, 'n3': 192}

# Make a grid
G = Grid(params)

# Derived FMKS metric parameter poly_norm
assert(compare(G.coords.poly_norm, 0.7578173169894967))

# Zone location in KS
X = G.coord(11,12,13)
rhp = np.squeeze(G.coords.ks_coord(X))
assert(compare(rhp, np.array([1.488590864996909, 0.7666458987406977, 0.4417864669110646])))

# Metric values:
gcov_computed = np.squeeze(G.coords.gcov(X))
gcov_example = np.array([[ 0.11428415,  1.65871321,  0.        , -0.5027359 ],
                         [ 1.65871321,  4.8045105 , -2.82071735, -1.41998137],
                         [ 0.        , -2.82071735, 66.60209297,  0.        ],
                         [-0.5027359 , -1.41998137,  0.        ,  1.71620473]])
assert(compare(gcov_computed, gcov_example))

gcon_computed = np.squeeze(G.coords.gcon(gcov_computed)) # TODO direct call from X
gcon_example = np.array([[-2.11428415e+00,  7.48549636e-01,  3.17024113e-02, -4.28728014e-17],
                         [ 7.48549636e-01,  1.98677175e-02,  8.41433249e-04,  2.35714628e-01],
                         [ 3.17024113e-02,  8.41433249e-04,  1.50501794e-02,  9.98293464e-03],
                         [-1.42468631e-17,  2.35714628e-01,  9.98293464e-03,  7.77710464e-01]])
assert(compare(gcon_computed, gcon_example))

# TODO FluidDump from existing data

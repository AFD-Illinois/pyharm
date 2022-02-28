#!/usr/bin/env python3

# Script to write an arbitrary dumpfile
# TODO WIP to write anything more than trivial zeros, needs BL->KS velocities and sensible starting config

import numpy as np
from pyharm import grid, units, coordinates
import pyharm.io.iharm3d as h5io
from pyharm.grmhd.init_tools import fourvel_to_prim, set_fourvel_t

# Problem-specific parameters
Ne_unit = 3.e-18
# GRRT prob 5
A = 1.e6
alpha = 0
height = 100./3
l0 = 1
a = 0.9

r_out = 50
n1 = 192
n2 = 128
n3 = 128

# Anything from the fake "GRMHD" run we're writing a supposed dump from.
t = 0
dt = 0
n_step = 0
n_dump = 0
params = {'dump_cadence': 5, 'cour': 0.9, 'gam': 4/3, 'tf': 0}

G = grid.make_some_grid('fmks', n1, n2, n3, a=a, r_out=r_out)
bl = coordinates.BL(met_params={'a': a})
P = np.zeros((8, n1, n2, n3))

r = G.coords.r(G.coord_all())
th = G.coords.th(G.coord_all())
phi = G.coords.phi(G.coord_all())

# Specific parameters
A = 1.e6
alpha = 0
height = 100./3
l0 = 1
a = 0.9

u = units.get_units(1, 1) # Code units

P[0] = Ne_unit * np.exp(-1/2 * ((r/10)**2 + (height * np.cos(th))**2)) / (u['RHO_unit']/(u['MP']+u['ME']))
P[1] = 0

l = (l0 / (1 + r*np.sin(th))) * (r*np.sin(th))**(1 + 0.5)
bl_gcon = bl.gcon()
ubar = np.sqrt(-1. / (bl_gcon[0][0] - 2. * bl_gcon[0][3] * l
                  + bl_gcon[3][3] * l * l))
bl_ucov = np.zeros_like()
bl_ucov[0] = -ubar
bl_ucov[1] = 0
bl_ucov[2] = 0
bl_ucov[3] = l * ubar

bl_ucon = G.raise_grid(bl_ucov)


set_fourvel_t(G.gcov, ucon)
fourvel_to_prim(G.gcon, ucon, P[2:5])

P[5] = 0
P[6] = 1
P[7] = 1

h5io.write_dump(params, G, P, t, dt, n_step, n_dump, "sample_dump.h5", dump_gamma=False)
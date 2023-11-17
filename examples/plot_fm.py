#!/usr/bin/env python3

import matplotlib.pyplot as plt

from pyharm.grmhd.tori import get_fm_torus_fluid_state, get_c_torus_fluid_state
from pyharm.grid import make_some_grid

import pyharm.plots.plot_dumps as pplt

g = make_some_grid('fmks', a=0.9375)
s = get_fm_torus_fluid_state(g, gamma=5./3)

print("r_in: {} r_max: {}".format(s['r_in'], s['r_max']))

sz = 15

fig, ax = plt.subplots(1,2,figsize=(10,5))
pplt.plot_xz(ax[0], s, 'rho', log=True, window=[-sz,sz,-sz,sz], vmin=1e-8, vmax=1.5)
pplt.plot_xy(ax[1], s, 'rho', log=True, window=[-sz,sz,-sz,sz], log_r=False, vmin=1e-8, vmax=1.5)
fig.savefig('fm_torus.png')

fig, ax = plt.subplots(1,2,figsize=(10,5))
pplt.plot_xz(ax[0], s, 'u', log=True, window=[-sz,sz,-sz,sz], vmin=1e-8, vmax=1)
pplt.plot_xy(ax[1], s, 'u', log=True, window=[-sz,sz,-sz,sz], log_r=False, vmin=1e-8, vmax=1)
fig.savefig('fm_torus_u.png')

s = get_c_torus_fluid_state(g, gamma=5./3)

fig, ax = plt.subplots(1,2,figsize=(10,5))
pplt.plot_xz(ax[0], s, 'rho', log=True, window=[-sz,sz,-sz,sz], vmin=1e-8, vmax=1)
pplt.plot_xy(ax[1], s, 'rho', log=True, window=[-sz,sz,-sz,sz], log_r=False, vmin=1e-8, vmax=1)
fig.savefig('c_torus.png')

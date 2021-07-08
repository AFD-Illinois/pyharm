#!/usr/bin/env python3

# Script to calculate how many zone-cycles are required to complete a simulation
# Usage: ./how_many_cycles.py n1 n2 n3 tf [nnodes [spin r_out system]]

import sys
import numpy as np
import pyHARM.grid as grid

n1 = int(sys.argv[1])
n2 = int(sys.argv[2])
n3 = int(sys.argv[3])
tf = float(sys.argv[4])

if len(sys.argv) > 5:
    nnodes = int(sys.argv[5])*6
else:
    nnodes = 6

if len(sys.argv) > 6:
    spin = float(sys.argv[6])
    r_out = float(sys.argv[7])
    system = sys.argv[8]
else:
    spin = 0.9375
    r_out = 1000.0
    system = 'fmks'

# ZCPS Estimates. Summit.
def kharma_perf(zones_per_node):
    return min(np.power(zones_per_node, 1/3)*15e6/150, 16e6)*nnodes
def grim_perf(zones_per_node):
    return min(np.power(zones_per_node, 1/3)*4.1e5/64, 4.1e5)*nnodes

print("Building grid: {}, a = {}, {}x{}x{} to r_out of {}".format(
      system, spin, n1, n2, n3, r_out))

G = grid.make_some_grid(system, n1, n2, n3, a=spin, r_out=r_out, cache_conn=False)

dt = G.dt_light()

print("Assuming dt: {}, total steps: {}".format(dt, tf/dt))
zc_total = tf/dt*n1*n2*n3
print("Total ZC estimate: {:g}".format(zc_total))

print("\nCampaign estimates:")
zones_per_node = n1*n2*n3/nnodes
perf = kharma_perf(zones_per_node)
time_h = zc_total/perf/60/60
print("KHARMA relative performance: {}%".format(perf/nnodes/17e6*100))
print("KHARMA: ", time_h, "hours,", time_h/12, "jobs", time_h*nnodes/6, "node-h")
perf = grim_perf(zones_per_node)
time_h = zc_total/perf/60/60
print("GRIM relative performance: {}%".format(perf/nnodes/3e5*100))
print("GRIM: ", time_h, "hours,", time_h/12, "jobs", time_h*nnodes/6, "node-h")

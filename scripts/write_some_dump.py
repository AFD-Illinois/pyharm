#!/usr/bin/env python3

# Script to write an arbitrary dumpfile

import numpy as np
import pyHARM.grid as grid
import pyHARM.h5io as h5io

# Sadly this is needed for the default dump() method in pyHARM
# Add anything else you want in the header to this dict and it'll show up.
# Subfolders should just be dicts within this one.
params = {'dump_cadence': 5.0}
a = 0.9375
r_out = 50
n1 = 192
n2 = 128
n3 = 128

G = grid.make_some_grid('fmks', n1, n2, n3, a=a, r_out=r_out)
P = np.zeros((8, n1, n2, n3))

print(P.shape)

t = 0
dt = 0

h5io.dump(params, G, P, t, dt, "sample_dump.h5", dump_gamma=False)

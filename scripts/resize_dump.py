#!/usr/bin/env python3
# Resize a dump file of any type into a KHARMA restart file with a globally refined grid

import sys
import numpy as np

import pyharm
from pyharm.grmhd.resize import resize
from pyharm.io.iharm3d import write_dump

dump = pyharm.load_dump(sys.argv[1], calc_derived=False, add_ghosts=True)

# TODO just return IharmDump object?
params_new, Gnew, Pnew = resize(dump.params, dump.grid, dump.prims, dump['n1']*2, dump['n2']*2, dump['n3']*2)

# params, G, P, t, dt, nstep, ndump, name
write_dump(params_new, Gnew, Pnew, 0.0, 0.0, 0, 0, "resized_dump.h5", out_type=np.float64)
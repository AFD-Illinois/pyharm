# Tries to load an HDF file, convert the prims to cons, and go back

import sys
# Faster norm?  Maybe?
from scipy.linalg import norm
import pyopencl as cl
import pyopencl.array as cl_array

# Change as necessary to point to pyHARM (TODO cross-platform script imports)
sys.path.append("../../pyHARM")
from defs import Loci

from u_to_p import U_to_P
from phys import prim_to_flux, get_state
from h5io import read_dump
from grid import Grid

P, params = read_dump("dump_00001200.h5")

params['ctx'] = cl.create_some_context()
print(params['ctx'])
params['queue'] = cl.CommandQueue(params['ctx'])

params['ng'] = 0
params['debug'] = True

G = Grid(params)

P = cl_array.to_device(params['queue'], P.copy())
D = get_state(params, G, P, Loci.CENT)
U = prim_to_flux(params, G, P, D, 0, Loci.CENT)

P_test, _ = U_to_P(params, G, U, 1.05*P)
print("Norm difference: ", norm((P - P_test).get()))

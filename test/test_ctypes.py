# Test ctypes u_to_p

import sys
import numpy as np

sys.path.append("../../pyHARM/")
from iharmc.iharmc import Iharmc
sys.path.append("../ana/")
from iharm_dump import IharmDump


ihc = Iharmc()

fname = "dump_00001200.h5"
dump = IharmDump(fname, add_cons=True)

prims0 = np.copy(dump.prims)
dump.prims *= 1.05

# Note this replaces the dump's primitives
ihc.u2p(dump)

print(prims0[:, 34, 45, 56])
print(dump.cons[:, 34, 45, 56])
print(dump.prims[:, 34, 45, 56])

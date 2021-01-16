#!/usr/bin/env python3

# This script converts KHARMA output to iharm3d-like IL HDF format,
# specifically a bunch of *.h5 files in a folder called dumps/
# Usage:
# $PATH_TO/kharma_convert.py *.phdf
# Note this places dumps/ and the output in the *current directory* when run.
# If you don't want that it should be simple to change.

import os
import sys
import pyHARM

os.makedirs("dumps", exist_ok=True)

for dumpname in sys.argv[1:]:
    dump = pyHARM.load_dump(dumpname)
    ndump = int(dumpname.split(".")[2]) # This assumes the usual Parthenon output format
    t = dump.params['dump_cadence'] * ndump
    # The zeros are dt and nstep.  KHARMA does not record those values at all
    pyHARM.io.ilhdf.write_dump(dump.params, dump.grid, dump.prims, t, 0, 0, ndump,
                                "dumps/"+dumpname.replace(".phdf", ".h5"))
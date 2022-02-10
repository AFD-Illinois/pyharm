#!/usr/bin/env python3

# Script to replicate the grid of a given runfile, exactly

import sys
import h5py
from pyHARM.grid import Grid
from pyHARM.io import gridfile
from pyHARM.io import iharm3d_header

# Read the header data of a given file to a dictionary
header = iharm3d_header.read_hdr(sys.argv[1])

# Generate a grid from the parameters in a standard
# header dictionary
G = Grid(header)

print("startx: ", G.startx)
print("stopx: ", G.stopx)
print("metric a, hslope:", G.coords.a, G.coords.hslope)

# Write a standard gridfile, like iharm3D would produce
# Won't match bit-for-bit due to gcon inversions,
# which are manual in iharm3D and np.linalg calls in
# pyHARM
gridfile.write(G, fname="grid.h5")

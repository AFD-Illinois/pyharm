# Read a HARM output file, and output a file with the zone center locations in BL coordinates

import sys
import h5py
import numpy as np

import pyHARM


dump = pyHARM.load_dump(sys.argv[1])

outf = h5py.File(sys.argv[2], "w")

# Get native coordinates of zone centers,
X = dump.grid.coord_all()
outf['X'] = X
# Convert r,th to KS==BL
r = dump.grid.coords.r(X)
outf['r'] = r
outf['th'] = dump.grid.coords.th(X)
# convert phi to BL coordinates
outf['phi'] = dump.grid.coords.phi(X) - dump['a'] / (r**2 - 2.*r + dump['a']**2) * r

# 4-vectors
bl = pyHARM.coordinates.BL({'a': dump['a']})
zero = np.zeros_like(r)
# Transform native->KS, then KS->BL
# The transformation matrix only depends on r[KS]==r[BL]
outf['ucon'] = dump.grid.dot(dump.grid.dot(dump['ucon'], dump.grid.coords.dxdX(dump.grid.coord_all())), bl.dXdx(np.array([zero,r,zero,zero])))
outf['Bcon'] = dump.grid.dot(dump.grid.dot(dump['bcon'], dump.grid.coords.dxdX(dump.grid.coord_all())), bl.dXdx(np.array([zero,r,zero,zero])))

outf.close()


import numpy as np
import h5py

from pyharm.defs import Loci

# TODO readers for iharm3d-style and ebhlight-style grids to Grid objects

# This does not need to be a class, it's one-and-done
def write(G, fname="grid.h5", astype=np.float32):
    """Dump a file containing grid zones.
    This will primarily be of archival use soon -- see grid.py, coordinates.py for
    a good way of reconstructing all common grids on the fly.
    """
    with h5py.File(fname, "w") as outf:
        x = G.coord_bulk(Loci.CENT).reshape(4, G.N[1], G.N[2], G.N[3])
        coords = G.coords

        outf['X'] = coords.cart_x(x).astype(astype)
        outf['Y'] = coords.cart_y(x).astype(astype)
        outf['Z'] = coords.cart_z(x).astype(astype)
        outf['r'] = coords.r(x).astype(astype)
        outf['th'] = coords.th(x).astype(astype)
        outf['phi'] = coords.phi(x).astype(astype)

        # Native coordinate output
        outf['X1'] = x[1].astype(astype)
        outf['X2'] = x[2].astype(astype)
        outf['X3'] = x[3].astype(astype)

        # Return only the CENT values, repeated over the N3 axis
        if G.NG > 0:
            b = slice(G.NG, -G.NG)
        else:
            b = slice(None, None)
        gcon3 = G.gcon[Loci.CENT.value, :, :, b, b, :].repeat(G.NTOT[3], axis=-1).transpose((2, 3, 4, 0, 1))
        gcov3 = G.gcov[Loci.CENT.value, :, :, b, b, :].repeat(G.NTOT[3], axis=-1).transpose((2, 3, 4, 0, 1))
        gdet3 = G.gdet[Loci.CENT.value, b, b, :].repeat(G.NTOT[3], axis=-1)
        lapse3 = G.lapse[Loci.CENT.value, b, b, :].repeat(G.NTOT[3], axis=-1)

        outf['gcon'] = gcon3.astype(astype)
        outf['gcov'] = gcov3.astype(astype)
        outf['gdet'] = gdet3.astype(astype)
        outf['lapse'] = lapse3.astype(astype)

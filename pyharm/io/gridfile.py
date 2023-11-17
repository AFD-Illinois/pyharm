__license__ = """
 File: gridfile.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020-2023, Ben Prather and AFD Group at UIUC
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import h5py

from pyharm.defs import Loci

__doc__ = \
"""Write "grid" files specifying geometry for e.g. visualization programs.
"""

# TODO readers for iharm3d-style and ebhlight-style grids to Grid objects
# TODO astype for vis writer

# This does not need to be a class, it's one-and-done
def write_grid(G, fname="grid.h5", astype=np.float32):
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
        gcon3 = G['gcon'][:, :, b, b, :].repeat(G.NTOT[3], axis=-1).transpose((2, 3, 4, 0, 1))
        gcov3 = G['gcov'][:, :, b, b, :].repeat(G.NTOT[3], axis=-1).transpose((2, 3, 4, 0, 1))
        gdet3 = G['gdet'][b, b, :].repeat(G.NTOT[3], axis=-1)
        lapse3 = G['lapse'][b, b, :].repeat(G.NTOT[3], axis=-1)

        outf['gcon'] = gcon3.astype(astype)
        outf['gcov'] = gcov3.astype(astype)
        outf['gdet'] = gdet3.astype(astype)
        outf['lapse'] = lapse3.astype(astype)

def write_vis_grid(G, outfname="grid.h5"):
    """Write a Grid object G to a bhlight-format grid file.
    Generally used for visualization with VisIt, to take advantage of XDMF and scripting from ebhlight.
    """

    outf = h5py.File(outfname, "w")

    # Cell coordinates
    x = G.coord_bulk(Loci.CENT).reshape(4, G.N[1], G.N[2], G.N[3])
    outf['Xharm'] = x.transpose(1,2,3,0)
    outf['Xcart'] = np.array([np.zeros([G.N[1],G.N[2],G.N[3]]), *G.coords.cart_coord(x)]).transpose(1,2,3,0)
    outf['Xbl'] = np.array([np.zeros([G.N[1],G.N[2],G.N[3]]), *G.coords.ks_coord(x)]).transpose(1,2,3,0)

    # Face coordinates
    xf = G.coord_bulk(mesh=True).reshape(4, G.N[1]+1, G.N[2]+1, G.N[3]+1)
    outf['XFharm'] = xf.transpose(1,2,3,0)
    outf['XFcart'] = np.array([np.zeros([G.N[1]+1,G.N[2]+1,G.N[3]+1]), *G.coords.cart_coord(xf)]).transpose(1,2,3,0)

    # Return corner values repeated over N3
    # TODO does bhlight really use corner values?
    if G.NG > 0:
        b = slice(None, None)
    else:
        b = slice(None, None)
    gamma = G['conn'][:, :, :, b, b, :].repeat(G.NTOT[3], axis=-1).transpose((3, 4, 5, 0, 1, 2))
    gcon3 = G['gcon_corner'][:, :, b, b, :].repeat(G.NTOT[3], axis=-1).transpose((2, 3, 4, 0, 1))
    gcov3 = G['gcov_corner'][:, :, b, b, :].repeat(G.NTOT[3], axis=-1).transpose((2, 3, 4, 0, 1))
    gdet3 = G['gdet_corner'][b, b, :].repeat(G.NTOT[3], axis=-1)
    lapse3 = G['lapse_corner'][b, b, :].repeat(G.NTOT[3], axis=-1)

    outf['Gamma'] = gamma
    outf['gcon'] = gcon3
    outf['gcov'] = gcov3
    outf['gdet'] = gdet3
    outf['alpha'] = lapse3

    dxdX = np.einsum("ij...,jk...->...ik", G.coords.dxdX_cart(x), G.coords.dxdX(x))
    outf['Lambda_h2cart_con'] = dxdX
    outf['Lambda_h2cart_cov'] = np.linalg.inv(dxdX)

    #TODO not used for VisIt but for completeness we should add:
    #Lambda_bl2cart_con       Dataset {32, 32, 1, 4, 4}
    #Lambda_bl2cart_cov       Dataset {32, 32, 1, 4, 4}
    #Lambda_h2bl_con          Dataset {32, 32, 1, 4, 4}
    #Lambda_h2bl_cov          Dataset {32, 32, 1, 4, 4}

    outf.close()

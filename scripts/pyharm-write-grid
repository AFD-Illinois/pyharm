#!/usr/bin/env python3

import click
import numpy as np
import h5py
from pyharm.defs import Loci
from pyharm import grid
from pyharm.grid import Grid
from pyharm import io

def write_bhlight_grid(G, outfname):
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
    xf = G.coord_bulk_mesh().reshape(4, G.N[1]+1, G.N[2]+1, G.N[3]+1)
    outf['XFharm'] = xf.transpose(1,2,3,0)
    outf['XFcart'] = np.array([np.zeros([G.N[1]+1,G.N[2]+1,G.N[3]+1]), *G.coords.cart_coord(xf)]).transpose(1,2,3,0)

    # Return only the CENT values, repeated over the N3 axis
    if G.NG > 0:
        b = slice(None, None)
    else:
        b = slice(None, None)
    locus = Loci.CORN.value
    gamma = G.conn[:, :, :, b, b, :].repeat(G.NTOT[3], axis=-1).transpose((3, 4, 5, 0, 1, 2))
    gcon3 = G.gcon[locus, :, :, b, b, :].repeat(G.NTOT[3], axis=-1).transpose((2, 3, 4, 0, 1))
    gcov3 = G.gcov[locus, :, :, b, b, :].repeat(G.NTOT[3], axis=-1).transpose((2, 3, 4, 0, 1))
    gdet3 = G.gdet[locus, b, b, :].repeat(G.NTOT[3], axis=-1)
    lapse3 = G.lapse[locus, b, b, :].repeat(G.NTOT[3], axis=-1)

    outf['Gamma'] = gamma
    outf['gcon'] = gcon3
    outf['gcov'] = gcov3
    outf['gdet'] = gdet3
    outf['alpha'] = lapse3

    dxdX = np.einsum("ij...,jk...->...ik", G.coords.dxdX_cartesian(x), G.coords.dxdX(x))
    outf['Lambda_h2cart_con'] = dxdX
    outf['Lambda_h2cart_cov'] = np.linalg.inv(dxdX)

    #TODO not used for VisIt but for completeness we should add:
    #Lambda_bl2cart_con       Dataset {32, 32, 1, 4, 4}
    #Lambda_bl2cart_cov       Dataset {32, 32, 1, 4, 4}
    #Lambda_h2bl_con          Dataset {32, 32, 1, 4, 4}
    #Lambda_h2bl_cov          Dataset {32, 32, 1, 4, 4}

    outf.close()

@click.command()
@click.argument('file_or_system', nargs=1)
@click.argument('spin', default=0.9375, nargs=1)
@click.argument('r_out', default=1000., nargs=1)
@click.argument('n1', default=192, nargs=1)
@click.argument('n2', default=128, nargs=1)
@click.argument('n3', default=128, nargs=1)
# Common options
@click.option('-o', '--outfile', default="grid.h5", help="Output file.")
@click.option('-vis', '--vis', is_flag=True, help="Whether to write to ebhlight/visualization format.")
def write_grid(file_or_system, spin, r_out, n1, n2, n3, outfile, vis):
    """Script to write files listing geometry information at each cell of a simulation.
    Used to avoid recalculating boring things like the metric, metric determinant, etc.

    \b
    Usage: write_grid.py FILE.{phdf,h5} 
    Writes a grid file using parameters for the simulation, given a dump of the simulation output

    \b
    Usage: write_grid.py SYSTEM SPIN R_OUT [N1 N2 N3]
    Writes a grid of given parameters and size:
    1. Coordinate SYSTEM in {fmks, mks, eks, ks, mks3, etc.}.
    2. SPIN of BH. Default 0.9375.
    3. R_OUT representing domain outer edge in r_g, default 1000.  Inner radius r_in will be determined
       in the same way as (modern! See relevant bug in ebhlight) HARM-like codes, putting 5 zones inside EH
    4. Grid size. Default 192x128x128
    5. Filename
    """

    # Read the header data of a given file to a dictionary
    try:
        header = io.read_hdr(file_or_system)
        # Generate a grid from the header
        G = Grid(header, cache_conn=True)

    except IOError:
        G = grid.make_some_grid(file_or_system, n1, n2, n3, a=spin, r_out=r_out, cache_conn=True)
        print("Could not read first argument as file, assuming grid parameters as arguments.")
        print("Building grid: {}, a = {}, {}x{}x{} to r_out of {}".format(
            file_or_system, spin, n1, n2, n3, r_out))

    print("Grid startx: ", G.startx)
    print("Grid stopx: ", G.stopx)
    print("Grid metric/coordinates {}, a={}, hslope={}".format(type(G.coords), G.coords.a, G.coords.hslope))
    print("Writing to file ", outfile)

    if vis:
        write_bhlight_grid(G, outfile)
    else:
        # Write a standard gridfile, like iharm3D would produce
        # Won't match bit-for-bit in gcon due to inversions,
        # but otherwise should.
        io.gridfile.write(G, fname=outfile)

if __name__ == "__main__":
    write_grid()
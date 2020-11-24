# File I/O.  Should be self-explanatory

import os
import h5py
import numpy as np

from pyHARM.defs import Loci
from pyHARM.grid import Grid
import pyHARM.parameters as parameters

from .phdf import phdf
from .hdr import *

def write_dump(params, G, P, t, dt, n_step, n_dump, fname, dump_gamma=True, out_type=np.float32):
    s = G.slices

    outf = h5py.File(fname, "w")

    write_hdr(params, outf)

    # Per-write_dump single variables
    outf['t'] = t
    outf['dt'] = dt
    outf['dump_cadence'] = params['dump_cadence']
    outf['full_dump_cadence'] = params['dump_cadence']
    outf['is_full_dump'] = 0
    outf['n_dump'] = n_dump
    outf['n_step'] = n_step

    # Arrays corresponding to actual data
    if G.NG > 0:
        outf["prims"] = np.einsum("p...->...p", P[s.allv + s.bulk]).astype(out_type)
    else:
        outf["prims"] = np.einsum("p...->...p", P).astype(out_type)

    # Extra in-situ calculations or custom debugging additions
    if "extras" not in outf:
        outf.create_group("extras")

    outf.close()


def read_dump(fname, params=None, **kwargs):
    """Read the header and primitives from a write_dump.
    No analysis or extra processing is performed
    @return P, params
    """

    # Not our job to read Parthenon files
    if ".phdf" in fname:
        if params is None:
            params = {}
        parameters.parse_parthenon_dat(fname.split("/")[-1].split(".")[0] + ".par", params)
        return read_dump_phdf(fname, params)

    with h5py.File(fname, "r") as infile:
        if params is not None:
            params.update(read_hdr(infile['/header']))
        else:
            params = read_hdr(infile['/header'])

        # Per-write_dump single variables.  TODO more?
        for key in ['t', 'dt', 'n_step', 'n_dump', 'is_full_dump', 'dump_cadence', 'full_dump_cadence']:
            if key in infile:
                params[key] = infile[key][()]

        P = _prep_array(infile['/prims'][()], **kwargs)

    return P, params

# Other readers, let users use as desired
def read_gamma(fname, **kwargs):
    with h5py.File(fname, "r") as infile:
        return _prep_array(infile['/gamma'][()], **kwargs)
def read_jcon(fname, **kwargs):
    with h5py.File(fname, "r") as infile:
        return _prep_array(infile['/jcon'][()], **kwargs)
def read_divb(fname, **kwargs):
    with h5py.File(fname, "r") as infile:
        return _prep_array(infile['/extras/divB'][()], **kwargs)
def read_fail_flags(fname, **kwargs):
    with h5py.File(fname, "r") as infile:
        if 'fail' in infile:
            return _prep_array(infile['/fail'][()], **kwargs)
        else:
            return _prep_array(infile['/extras/fail'][()], **kwargs)
def read_floor_flags(fname, **kwargs):
    with h5py.File(fname, "r") as infile:
        return _prep_array(infile['/extras/fixup'][()], **kwargs)

def _prep_array(arr, zones_first=False, as_double=False):
    """Re-order and optionally up-convert an array from a file,
    to put it in usual pyHARM order/format
    """
    # Reverse indices on vectors, since most pyHARM tooling expects p,i,j,k
    # See iharm_dump for analysis interface that restores i,j,k,p order
    if (not zones_first) and (len(arr.shape) > 3):
        arr = np.einsum("...m->m...", arr)

    # Also upconvert to doubles if necessary
    # TODO could add other types?  Not really necessary yet
    if as_double:
        arr = arr.astype(np.float64)
    
    return arr

def read_dump_phdf(fname, params):
    f = phdf(fname)

    # Hope that Grid fills the implicit parameters how KHARMA does
    G = Grid(params)

    # This is the "true" number of zones for the sim, but doesn't concern us
    params['ng'] = f.NGhost
    # This is the number present in the file
    ng_f = f.IncludesGhost * f.NGhost
    # This is the number we should include on output: i.e. present & requested
    if 'include_ghost' in params:
        ng_ix = params['include_ghost'] * ng_f
    else:
        params['include_ghost'] = False
        ng_ix = 0
    
    # Trivial dimensions don't have ghosts
    if params['n2'] == 1:
        ng_iy = 0
    else:
        ng_iy = ng_ix

    if params['n3'] == 1:
        ng_iz = 0
    else:
        ng_iz = ng_ix

    params['startx1'] = G.startx[1]
    params['startx2'] = G.startx[2]
    params['startx3'] = G.startx[3]
    params['dx1'] = dx = G.dx[1]
    params['dx2'] = dy = G.dx[2]
    params['dx3'] = dz = G.dx[3]

    # Lay out the blocks and determine total mesh size
    bounds = []
    for ib in range(f.NumBlocks):
        bb = f.BlockBounds[ib]
        # Internal location of the block i.e. starting/stopping indices in the final, big mesh
        bound = [int((bb[0]+dx/2 - G.startx[1])/dx)-ng_ix, int((bb[1]+dx/2 - G.startx[1])/dx)+ng_ix,
                 int((bb[2]+dy/2 - G.startx[2])/dy)-ng_iy, int((bb[3]+dy/2 - G.startx[2])/dy)+ng_iy,
                 int((bb[4]+dz/2 - G.startx[3])/dz)-ng_iz, int((bb[5]+dz/2 - G.startx[3])/dz)+ng_iz]
        bounds.append(bound)

    # Optionally allocate enough space for ghost zones
    P = np.zeros((8, G.NTOT[1]+2*ng_ix, G.NTOT[2]+2*ng_iy, G.NTOT[3]+2*ng_iz))

    # Read blocks into their slices of the full mesh
    for ib in range(f.NumBlocks):
        b = bounds[ib]
        # Exclude ghost zones if we don't want them
        if ng_ix == 0 and ng_f != 0:
            if params['n2'] == 1:
                o = [ng_f, -ng_f, None, None, None, None]
            elif params['n3'] == 1:
                o = [ng_f, -ng_f, ng_f, -ng_f, None, None]
            else:
                o = [ng_f, -ng_f, ng_f, -ng_f, ng_f, -ng_f]
        else:
            o = [None, None, None, None, None, None]
        # False == don't flatten into 1D array
        #print("Bound ", b)
        P[:, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = f.Get('c.c.bulk.prims', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1],:].transpose(3,2,1,0)

    return (P, params)


# For cutting on time without loading everything
def get_dump_time(fname):
    dfile = h5py.File(fname, 'r')

    if 't' in dfile.keys():
        t = dfile['t'][()]
    else:
        t = 0

    dfile.close()
    return t

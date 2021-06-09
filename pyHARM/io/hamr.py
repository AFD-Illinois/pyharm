# i/o for H-AMR files

import os
import h5py
import numpy as np

from pyHARM.defs import Loci
from pyHARM.grid import Grid
import pyHARM.parameters as parameters

def read_dump(fname, params=None, add_ghosts=False, **kwargs):
    """Read the header and primitives from a write_dump.
    No analysis or extra processing is performed
    @return P, params
    """

    with h5py.File(fname, "r") as infile:
        if params is None:
            params = {}

        # Per-write_dump single variables
        # dt? n_nstep? n _dump?
        # What about dscale?
        for hdr_key, par_key in [('t','t'), ('gam', 'gam'), ('dscale', 'dscale'),
                    ('a','a'), ('hslope', 'hslope'), ('N1','n1'), ('N2','n2'), ('N3','n3'),
                    ('R0','r0'), ('Rin','r_in'), ('Rout','r_out')]:
            if hdr_key in infile.attrs:
                params[par_key] = infile.attrs[hdr_key]
        
        params['dx1'], params['dx2'], params['dx3'] = infile.attrs['dx']

        # Translate header variables.  Taken from ipole.
        params['startx1'] = infile.attrs['startx'][0] - params['dx1']/2
        params['startx2'] = infile.attrs['startx'][1] - params['dx2']/2
        params['startx3'] = 0 # infile.attrs['startx'][2]

        # Translate from hamr x2 \in (-1, 1) -> mks x2 \in (0, 1)
        params['startx2'] = (params['startx2'] + 1)/2.
        #params['stopx2'] = (stopx[2] + 1)/2.
        params['dx2'] /= 2
        
        params = parameters._fix(params)

        n1, n2, n3 = params['n1'], params['n2'], params['n3']
        P = np.zeros((8, n1, n2, n3))
        P[0,:,:,:] = infile['RHO'][()].reshape((n1, n2, n3))
        P[1,:,:,:] = infile['UU'][()].reshape((n1, n2, n3))
        P[2,:,:,:] = infile['U1'][()].reshape((n1, n2, n3))
        P[3,:,:,:] = infile['U2'][()].reshape((n1, n2, n3))
        P[4,:,:,:] = infile['U3'][()].reshape((n1, n2, n3))
        P[5,:,:,:] = infile['B1'][()].reshape((n1, n2, n3))
        P[6,:,:,:] = infile['B2'][()].reshape((n1, n2, n3))
        P[7,:,:,:] = infile['B3'][()].reshape((n1, n2, n3))

        P = _prep_array(P, **kwargs)

    return P, params

# Other readers, let users use as desired
# This is Ucon0 in file
def read_gamma(fname, **kwargs):
    return None

# These will likely never be present
def read_jcon(fname, **kwargs):
    return None
def read_divb(fname, **kwargs):
    return None
def read_fail_flags(fname, **kwargs):
    return None
def read_floor_flags(fname, **kwargs):
    return None

def _prep_array(arr, as_double=False, zones_first=False, add_ghosts=False):
    """Re-order and optionally up-convert an array from a file,
    to put it in usual pyHARM order/format
    """
    # Upconvert to doubles if necessary
    # TODO could add other types?  Not really necessary yet
    if as_double:
        arr = arr.astype(np.float64)
    
    return arr


# For cutting on time without loading everything
def get_dump_time(fname):
    # TODO use with?
    dfile = h5py.File(fname, 'r')

    if 't' in dfile.attrs.keys():
        t = dfile.attrs['t']
    else:
        t = 0

    dfile.close()
    return t
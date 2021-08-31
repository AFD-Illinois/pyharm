# i/o for KORAL files

import os
import h5py
import numpy as np

from pyHARM.defs import Loci
from pyHARM.grid import Grid
import pyHARM.parameters as parameters
from pyHARM.io.hdr import read_hdr

def read_dump(fname, params=None, add_ghosts=False, **kwargs):
    """Read the header and primitives from a write_dump.
    No analysis or extra processing is performed
    @return P, params
    """

    with h5py.File(fname, "r") as infile:
        if params is not None:
            params.update(read_hdr(infile['/header']))
        else:
            params = read_hdr(infile['/header'])

        n1, n2, n3 = params['n1'], params['n2'], params['n3']
        P = np.zeros((8, n1, n2, n3))
        quants = infile['quants']
        P[0,:,:,:] = quants['rho'][()]
        P[1,:,:,:] = quants['uint'][()]
        P[2,:,:,:] = quants['U1'][()]
        P[3,:,:,:] = quants['U2'][()]
        P[4,:,:,:] = quants['U3'][()]
        P[5,:,:,:] = quants['B1'][()]
        P[6,:,:,:] = quants['B2'][()]
        P[7,:,:,:] = quants['B3'][()] #.reshape((n1, n2, n3))

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
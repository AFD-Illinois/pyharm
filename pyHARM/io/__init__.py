
# Wrap functions from individual io modules by detecting filenames
from glob import glob

import pyHARM.io.ilhdf as ilhdf
import pyHARM.io.kharma as kharma
import pyHARM.io.harm2d as harm2d
import pyHARM.io.hamr as hamr

# Also import i/o for other filetypes directly
from pyHARM.io.misc import *
from pyHARM.io.gridfile import *

def get_fnames(path):
    files = np.sort(glob(os.path.join(path, "dump_*.h5")))
    if len(files) == 0:
        files = np.sort(glob(os.path.join(path, "*out*.phdf")))
    if len(files) == 0:
        files = np.sort(glob(os.path.join(path, "dump*")))
    if len(files) == 0:
        files = np.sort(glob(os.path.join(path, "dump_*.h5")))
    if len(files) == 0:
        files = np.sort(glob(os.path.join(path, "dump_*.hdf5")))
    if len(files) == 0:
        raise FileNotFoundError("No dump files found at {}".format(path))
    return files

def get_filter(fname):
    """Choose an importer based on what we know of file contents, or failing that, names
    Purposefully a bit dumb, just trusts the filename
    """
    if ".phdf" in fname:
        return kharma
    elif ".h5" in fname:
        if 'header' in h5py.File(fname).keys():
            return ilhdf
        else:
            return kharma
    elif ".hdf5" in fname:
        return hamr
    elif "dump" in fname:
        # HARM ASCII files are *usually* named "dump" with a number
        return harm2d
    else:
        print("Guessing filetype harm2d!")
        return harm2d

def read_dump(fname, params=None):
    """Try to automatically read a dump file by guessing the filetype.
    See individual implementations for more options
    """
    my_filter = get_filter(fname)
    return my_filter.read_dump(fname, params=params)

def read_hdr(fname, params=None):
    """Try to automatically read a dump's header by guessing the filetype.
    See individual implementations for more options
    """
    my_filter = get_filter(fname)
    return my_filter.read_hdr(fname, params=params)

def get_dump_time(fname):
    """Try to automatically read a dump's header by guessing the filetype.
    See individual implementations for more options
    """
    my_filter = get_filter(fname)
    return my_filter.get_dump_time(fname)

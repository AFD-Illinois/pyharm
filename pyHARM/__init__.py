# Load a selection of useful functions and aliases to serve as a high-level interface

import h5py

from .ana.iharm_dump import IharmDump
from .ana.reductions import *
from .ana.variables import pretty

from .ana import plot

from .defs import Loci, Var

def load_dump(fname, **kwargs):
    """Wrapper for creating a new IharmDump object from the given file
    See pyHARM/ana/iharm_dump.py
    """
    return IharmDump(fname, **kwargs)
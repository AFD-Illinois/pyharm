# Load a selection of useful functions and aliases to serve as a high-level interface

from .fluid_dump import FluidDump
from .reductions import *
from .plots.pretty import pretty

from .defs import Loci

def load_dump(fname, **kwargs):
    """Wrapper for creating a new IharmDump object from the given file
    See pyharm/ana/iharm_dump.py
    """
    return FluidDump(fname, **kwargs)

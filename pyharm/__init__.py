# Load a selection of useful functions and aliases to serve as a high-level interface

from pyharm.ana_results import AnaResults
from .fluid_dump import FluidDump
from .ana.reductions import *
from .plots.pretty import pretty

from .defs import Loci

def load_dump(fname, **kwargs):
    """Wrapper to create a new FluidDump object using the given file
    See pyharm/fluid_dump.py.
    """
    return FluidDump(fname, **kwargs)

def load_results(fname, **kwargs):
    """Wrapper to read diagnostic output or results of reductions
    """
    return AnaResults(fname, **kwargs)

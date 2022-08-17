# Load a selection of useful functions and aliases to serve as a high-level interface

from pyharm.ana_results import AnaResults
from .fluid_dump import FluidDump
from .ana.reductions import *
from .plots.pretty import pretty

from .defs import Loci

from .ana_results import load_result, load_results, load_results_glob

def load_dump(fname, **kwargs):
    """Wrapper to create a new FluidDump object using the given file
    See pyharm/fluid_dump.py.
    """
    return FluidDump(fname, **kwargs)

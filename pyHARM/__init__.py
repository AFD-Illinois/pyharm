# Load a selection of useful functions and aliases to serve as a high-level interface

from .ana.iharm_dump import IharmDump
from .ana.reductions import *
from .ana.variables import pretty

def load_dump(fname):
    return IharmDump(fname)
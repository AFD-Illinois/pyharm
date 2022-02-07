# Object representing an iharm dump file
# Contains fluid state, and definitions of some common derived fields

import copy
import numpy as np

from pyHARM.grid import Grid

import pyHARM.io as io
from pyHARM.io.interface import DumpFile

import pyHARM.variables as vars
from pyHARM.grmhd.b_field import divB
from pyHARM.units import get_units

class FluidDump:
    """Read and cache data from a fluid dump file in HARM HDF5 format, and allow accessing
    various derived variables directly.
    """

    def __init__(self, fname, tag="", ghost_zones=False, grid_cache=True, cache_conn=False, units=None):
        """Attach the fluid dump file 'fname' and make its contents accessible like a dictionary.  For a list of some
        variables and properties accessible this way, see 

        :param fname: file name or path to dump.
        :param ghost_zones: Load ghost zones when reading from a dump file
        :param grid_cache: Cache geometry values in the grid file.  These are *not* yet automatically added,
                           so keep this True unless plotting a very simple variable.
        :param cache_conn: Cache the connection coefficients at zone centers. Memory-intensive, rarely needed
        :param units: a 'Units' object representing a physical scale for the dump (density M_unit and BH mass MBH)
        """
        self.fname = fname
        if tag == "":
            self.tag = fname
        else:
            self.tag = tag
        self.units = units

        # Choose an importer based on what we know of filenames
        self.reader = io.file_reader(fname, ghost_zones=ghost_zones)
        self.params = self.reader.params
        self.cache = {}
        # These will only be modified directly, by the slice "constructor"
        self.slice = ()
        self.grid = Grid(self.params, caches=grid_cache, cache_conn=cache_conn)

    def set_units(self, MBH, M_unit):
        """Associate a scale & units with this dump, for calculating scale-dependent quantities in CGS"""
        self.units = get_units(MBH, M_unit, gam=self.params['gam'])

    def __getitem__(self, key):
        """Get any of a number of different things from the backing dump file,
        or from a cached version.

        Also allows slicing FluidDump objects to get just a section, and read/operate on that section thereafter
        Supports only a small subset of slicing operations, must pass a list/tuple of some sort & not None
        Also note this means no requesting lists of variables at once. I have no idea why you'd want that.
        Just, don't.
        """
        # Allow slicing FluidDump objects to get just a section, and read/operate on that section thereafter
        # We'll only want multi-dimensional slices, not elements
        # Also note this means no requesting lists of variables at once. I have no idea why you'd want that.
        # Just, don't.
        if type(key) in (list, tuple):
            slc = key
            out = copy.copy(self)
            for c in out.cache:
                out.cache[c] = out.cache[c][slc]
            out.grid = out.grid[slc]
            out.slice = slc
            return out

        # Return things from the cache if we can
        elif key in self.cache:
            return self.cache[key]
        elif key in self.params:
            return self.params[key]

        # Otherwise run functions and cache the result
        # Putting this before reading lets us translate & standardize reads/caches
        elif key in vars.fns_dict:
            self.cache[key] = vars.fns_dict[key](self)
            return self.cache[key]

        # Return coordinates and things from the grid
        elif key in self.grid.can_provide:
            return self.grid[key]

        # Prefixes for a few common 1:1 math operations.
        # Most math should be done by reductions.py
        # Don't bother to cache these, they aren't intensive to calculate
        elif key[:5] == "sqrt_":
            return np.sqrt(self[key[5:]])
        elif key[:4] == "abs_":
            return np.abs(self[key[4:]])
        elif key[:4] == "log_":
            return np.log10(self[key[4:]])
        elif key[:3] == "ln_":
            return np.log(self[key[3:]])

        # TODO transformed full vectors, with e.g. 'ucon_ks'
        # Return vector components
        elif key[-2:] == "_0" or key[-2:] == "_1" or key[-2:] == "_2" or key[-2:] == "_3":
            return self[key[0]+"cov"][int(key[-1])]
        elif key[-2:] == "^0" or key[-2:] == "^1" or key[-2:] == "^2" or key[-2:] == "^3":
            return self[key[0]+"con"][int(key[-1])]
        # Return transformed vector components
        # TODO Cartesian forms, move the complexity here to grid
        elif key[-2:] == "_t" or key[-2:] == "_r" or key[-3:] == "_th" or key[-4:] == "_phi":
            return np.einsum("i...,ij...->j...",
                                self[key[0]+"cov"],
                                self.grid.coords.dxdX(self.grid.coord_all())
                            )[["t", "r", "th", "phi"].index(key.split("_")[-1])]
        elif key[-2:] == "^t" or key[-2:] == "^r" or key[-3:] == "^th" or key[-4:] == "^phi":
            return np.einsum("i...,ij...->j...",
                                self[key[0]+"con"],
                                self.grid.coords.dXdx(self.grid.coord_all())
                            )[["t", "r", "th", "phi"].index(key.split("^")[-1])]

        else:
            # Read things that we haven't cached and absolutely can't calculate
            # The reader keeps its own cache, so we don't add its items to ours
            if "flag" in key:
                out = self.reader.read_var(key, astype=np.int32, slc=self.slice)
            else:
                out = self.reader.read_var(key, astype=np.float32, slc=self.slice)
            if out is None:
                raise ValueError("FluidDump cannot find or compute {}".format(key))
            else:
                return out

        raise RuntimeError("Reached the end of FluidDump __getitem__, returning None")



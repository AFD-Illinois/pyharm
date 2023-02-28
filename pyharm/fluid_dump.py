__license__ = """
 File: fluid_dump.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020-2022, AFD Group at UIUC
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import copy
import numpy as np

from .defs import Loci
from .grid import Grid

from . import io
from . import variables
from .grmhd.b_field import divB
from .units import get_units

__doc__ = \
"""This file exclusively defines an object representing a GRMHD code snapshot or "dump" file.
The object serves as a cohesive interface for any data or calculations concerning the fluid state data.
"""

def load_dump(fname, **kwargs):
    """Wrapper to create a new FluidDump object using the given file
    """
    return FluidDump(fname, **kwargs)

class FluidDump:
    """Read and cache data from a fluid dump file in any supported format, and allow accessing
    various derived properties directly.
    """

    def __init__(self, fname, tag="", ghost_zones=False, grid_cache=True, cache_conn=False, units=None, add_grid=True, params=None):
        """Attach the fluid dump file 'fname' and make its contents accessible like a dictionary.  For a list of some
        variables and properties accessible this way, see the README.

        Fluid dumps can be sliced like arrays!  That is, ``dump[i,j,k]['var_name']`` will read or compute ``'var_name'`` only for the
        particular index in question, and similarly for slices of any size (e.g., 2D slices for plots).  This is
        *tremendously useful*, so remember to slice first to save time if efficiency is important.  Note, though,
        that slicing preserves the dimensionality -- that is, ``dump[i,j,k][var].shape`` will be (1,1,1).  Being able to assume
        every dump has three dimensions makes a lot of internal logic easier.  If you need 2D arrays, just use
        ``np.squeeze``, or the internal functions ``flatten_xz`` and ``flatten_xy``, which return common 2D slices or averages.

        Also, note that slicing does not support strides, and that slices may be *views* rather than *copies* --
        if you're going to modify array contents yourself within a slice (for some reason...), it may affect the global array.
        Generally this will just behave how you want, but it can be confusing if you're really digging around.  If you have the memory,
        you can use ``copy.copy`` or ``copy.deepcopy`` to be certain.

        :param fname: file name or path to dump
        :param tag: any string, usually long name of dump/model for plotting
        :param ghost_zones: Load ghost zones when reading from a dump file
        :param grid_cache: Cache geometry values in the grid file.  These are *not* yet automatically added,
                           so keep this True unless plotting a very simple variable
        :param cache_conn: Cache the connection coefficients at zone centers. Default off as memory-intensive and rarely needed
        :param units: a 'Units' object representing a physical scale for the dump (density M_unit and BH mass MBH)
        :param add_grid: Whether to construct a Grid object at all.  Only used for copy construction.
        :param params: dictionary of parameters. Only used for copy construction.
        """
        self.fname = fname
        if tag == "":
            self.tag = fname
        else:
            self.tag = tag
        self.units = units

        # Choose an importer based on what we know of filenames
        self.reader = io.file_reader(fname, params=params, ghost_zones=ghost_zones)
        if params is None:
            self.params = self.reader.params
        else:
            self.params = params
        self.cache = {}
        self.slice = ()
        if add_grid:
            self.grid = Grid(self.params, caches=grid_cache, cache_conn=cache_conn)
        else:
            self.grid = None

    def __del__(self):
        # Try to clean up what we can. Anything that may possibly not be a simple ref
        for cache in ('cache', 'units', 'params', 'grid'):
            if cache in self.__dict__:
                del self.__dict__[cache]

    def set_units(self, MBH, M_unit):
        """Associate a scale & units with this dump, for calculating scale-dependent quantities in CGS.
        :param MBH: Black hole mass in solar masses
        :param M_unit: Density unit in grams, as fit by imaging with e.g. ``ipole``

        Note this function will not *change* anything -- rather it *adds* a dict 'units' and a number of keys:
        'M_unit', 'RHO_unit', 'T_unit', etc.  See ``units.py`` for definitions.
        """
        self.units = get_units(MBH, M_unit, gam=self.params['gam'])

    def __getitem__(self, key):
        """Get any of a number of different things from the backing dump file, or from a cached version.
        The full list of keys is covered in depth in the documentation at :ref:`keys`.

        Also allows slicing FluidDump objects to get just a section, and read/operate on just that section
        thereafter. This supports only a small subset of slicing operations:  you must pass a tuple of three
        elements, all of which must either be integers or slice objects (not None).
        Due to overloading, it is thus impossible to allow requesting lists of variables at once.
        I have no idea why you'd want that.  Just, don't.
        """
        if type(key) in (list, tuple):
            slc = key
            # TODO handle further slicing after this is a 2D object?
            relevant = [False, False, False]
            new_slc = list(slc)
            for i in range(3):
                if isinstance(slc[i], slice):
                    new_slc[i] = slc[i]
                else:
                    new_slc[i] = slice(slc[i], slc[i]+1) # For gauging relevance later
                relevant[i] = ((new_slc[i].start is not None) or (new_slc[i].stop is not None))

            if not (relevant[0] or relevant[1] or relevant[2]):
                return self

            # TODO somehow proper copy constructor
            slc = tuple(new_slc)
            #print("FluidDump slice copy: ", self.cache, key)
            out = FluidDump(self.fname, add_grid=False, params=self.params, units=self.units)
            #out = copy.deepcopy(self) # In case this proves faster
            for c in self.cache:
                out.cache[c] = self.cache[c][(Ellipsis,) + slc]
            if self.grid is not None:
                out.grid = self.grid[slc]
            out.slice = slc
            return out

        # Return things from the cache if we can
        elif key in self.cache:
            return self.cache[key]
        elif key in self.params:
            return self.params[key]
        elif self.units is not None and key in self.units:
            return self.units[key]

        # Otherwise run functions and cache the result
        # Putting this before reading lets us translate & standardize reads/caches
        elif key in variables.fns_dict:
            self.cache[key] = variables.fns_dict[key](self)
            return self.cache[key]

        # Return coordinates and things from the grid
        # Default to centers when returning multi-location vars, to avoid location madness
        # TODO allow _mesh versions?
        elif self.grid is not None and key in self.grid:
            if key in ('gcon', 'gcov', 'gdet', 'lapse'):
                return self.grid[key][Loci.CENT.value]
            else:
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
        elif key[:4] == "inv_":
            return 1/self[key[4:]]
        elif key[:4] == "neg_":
            return -self[key[4:]]

        # Return MHD tensor components: don't cache
        elif ((key[-2:] == "_0" or key[-2:] == "_1" or key[-2:] == "_2" or key[-2:] == "_3")
              and (key[-4:-2] == "_0" or key[-4:-2] == "_1" or key[-4:-2] == "_2" or key[-4:-2] == "_3")):
            i, j = int(key[-3]), int(key[-1])
            if key[-5:-4] == "T":
                return variables.T_cov(self, i, j)
            elif key[-5:-4] == "F":
                return variables.F_cov(self, i, j)

        elif ((key[-2:] == "_0" or key[-2:] == "_1" or key[-2:] == "_2" or key[-2:] == "_3")
              and (key[-4:-2] == "^0" or key[-4:-2] == "^1" or key[-4:-2] == "^2" or key[-4:-2] == "^3")):
            i, j = int(key[-3]), int(key[-1])
            if key[-5:-4] == "T":
                return variables.T_mixed(self, i, j)
            elif key[-7:-4] == "TEM":
                return variables.TEM_mixed(self, i, j)
            elif key[-9:-4] == "TPAKE":
                return variables.TPAKE_mixed(self, i, j)
            elif key[-7:-4] == "TEN":
                return variables.TEN_mixed(self, i, j)
            elif key[-7:-4] == "TFl":
                return variables.TFl_mixed(self, i, j)

        elif ((key[-2:] == "^0" or key[-2:] == "^1" or key[-2:] == "^2" or key[-2:] == "^3")
              and (key[-4:-2] == "^0" or key[-4:-2] == "^1" or key[-4:-2] == "^2" or key[-4:-2] == "^3")):
            i, j = int(key[-3]), int(key[-1])
            if key[-5:-4] == "T":
                return variables.T_con(self, i, j)
            elif key[-5:-4] == "F":
                return variables.F_con(self, i, j)

        # Return vector components: do cache
        elif key[-2:] == "_0" or key[-2:] == "_1" or key[-2:] == "_2" or key[-2:] == "_3":
            return self[key[:-2]+"cov"][int(key[-1])]
        elif key[-2:] == "^0" or key[-2:] == "^1" or key[-2:] == "^2" or key[-2:] == "^3":
            return self[key[:-2]+"con"][int(key[-1])]

        # Return transformed vector components
        elif key[-2:] == "_t" or key[-2:] == "_r" or key[-3:] == "_th" or key[-4:] == "_phi":
            return self[key[0]+"cov_base"][["t", "r", "th", "phi"].index(key.split("_")[-1])]
        elif key[-2:] == "^t" or key[-2:] == "^r" or key[-3:] == "^th" or key[-4:] == "^phi":
            return self[key[0]+"con_base"][["t", "r", "th", "phi"].index(key.split("^")[-1])]
        elif key[-2:] == "_x" or key[-2:] == "_y" or key[-2:] == "_z":
            return self[key[0]+"cov_cart"][["t", "x", "y", "z"].index(key.split("_")[-1])]
        elif key[-2:] == "^x" or key[-2:] == "^y" or key[-2:] == "^z":
            return self[key[0]+"con_cart"][["t", "x", "y", "z"].index(key.split("^")[-1])]

        # Return an array of the correct size filled with just zero or one
        # Don't cache these
        # TODO avoid file read?
        elif key in ('zero', '0'):
            return np.zeros_like(self['rho'])
        elif key in ('one', '1'):
            return np.ones_like(self['rho'])
        else:
            # Read things that we haven't cached and absolutely can't calculate
            # The reader keeps its own cache, so we don't add its items to ours
            if "flag" in key:
                out = self.reader.read_var(key, astype=np.int32, slc=self.slice)
            else:
                # TODO Option for double
                out = self.reader.read_var(key, astype=np.float64, slc=self.slice)
            if out is None:
                raise ValueError("FluidDump cannot find or compute {}".format(key))
            else:
                return out

        raise RuntimeError("Reached the end of FluidDump.__getitem__, should have returned a value!")



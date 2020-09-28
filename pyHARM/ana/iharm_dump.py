# Object representing an iharm dump file
# Contains fluid state, and definitions of some common derived fields

import os
import sys
import numpy as np
import h5py

from pyHARM.defs import Loci
from pyHARM.h5io import read_dump, read_jcon, read_fail_flags, read_floor_flags
try:
    from pyHARM import phys
    can_use_cl = True
except ImportError:
    #print("Loading IharmDump without OpenCL support.")
    can_use_cl = False
from pyHARM.pure_np import phys as np_phys
from pyHARM.grid import Grid

import pyHARM.ana.variables as vars


class IharmDump:
    """Read and cache data from a fluid dump file in HARM HDF5 format, and allow accessing
    various derived variables directly.
    """

    def __init__(self, fname, params=None, calc_cons=False, calc_derived=False,
                 add_jcon=False, add_floors=False, add_fails=False,
                 zones_first=False, lock=None):
        """Read the HDF5 file 'fname' into memory, and pre-calculate/cache useful variables
        @param calc_cons: calculate the conserved variables U, i.e. run 'prim_to_flux(...,0)' from HARM
        @param calc_derived: calculate the derived 4-vectors u, b and fluid Lorentz factor gamma

        @param add_jcon: Read the current jcon from the file if it exists, fail if it doesn't
        @param add_floors: Read the applied floors bitflag from the file if it exists, fail if it doesn't
        @param add_fails: Read the inversion failures bitflag from the file if it exists, fail if it doesn't

        @param zones_first: keep arrays and vectors in i,j,k,p order rather than native p,i,j,k, usually
        for immediate output.  Breaks lots of physics code!
        @param lock: mutex lock for any OpenCL context being passed in params.
        """
        # TODO allow adding gamma, U from file vs calculating them
        self.fname = fname
        if params is None:
            params = {}

        if 'queue' in params and can_use_cl:
            self._use_cl = True
        else:
            self._use_cl = False

        P, params = read_dump(fname, params=params)

        if add_jcon:
            self.jcon = read_jcon(fname)
        if add_fails:
            self.fails = read_fail_flags(fname)
        if add_floors:
            self.floors = read_floor_flags(fname)

        self.header = self.params = params

        if ('include_ghost' not in self.header) or (not self.header['include_ghost']):
            self.header['ng'] = 0
        G = Grid(self.header)
        self.grid = G
        if zones_first:
            self.gcon = G.gcon[Loci.CENT.value].transpose(2, 3, 0, 1)
            self.gcov = G.gcov[Loci.CENT.value].transpose(2, 3, 0, 1)
        else:
            self.gcon = G.gcon[Loci.CENT.value]
            self.gcov = G.gcov[Loci.CENT.value]
        self.gdet = G.gdet[Loci.CENT.value]
        self.lapse = G.lapse[Loci.CENT.value]

        self.N1 = G.NTOT[1]
        self.N2 = G.NTOT[2]
        self.N3 = G.NTOT[3]

        if calc_derived or calc_cons:
            # Derived variables, with or without OCL
            # Calculating w/temp P avoids issues with the variable reordering
            if self._use_cl:
                if lock is not None:
                    lock.acquire()
                
                P_tmp = P.astype(np.float64)
                D = phys.get_state(self.header, G, P_tmp)
                self.gamma = phys.mhd_gamma_calc(self.header['queue'], G, P_tmp).get()
                if calc_cons:
                    U = phys.prim_to_flux(self.header, G, P_tmp, D=D).get()

                self.ucon = D['ucon'].get()
                self.ucov = D['ucov'].get()
                self.bcon = D['bcon'].get()
                self.bcov = D['bcov'].get()
                
                del P_tmp,D
                
                if lock is not None:
                    lock.release()

            else:
                D = np_phys.get_state(self.header, G, P)
                self.gamma = np_phys.mhd_gamma_calc(G, P)
                if calc_cons:
                    U = np_phys.prim_to_flux(self.header, self.grid, P, D=D)

                self.ucon = D['ucon']
                self.ucov = D['ucov']
                self.bcon = D['bcon']
                self.bcov = D['bcov']

                del D
        
        # We're done using the grid's openCL, and it causes MemoryErrors in multiprocess code, so...
        self.grid.use_ocl = False

        self._zones_first = zones_first
        if zones_first and (calc_derived or calc_cons):
            # Turn around all our lovely zones-last arrays
            self.prims = np.ascontiguousarray(np.einsum("p...->...p", P))
            if calc_cons:
                self.cons = np.einsum("p...->...p", U)
            if calc_derived or calc_cons:
                self.ucon = np.einsum("p...->...p", self.ucon)
                self.ucov = np.einsum("p...->...p", self.ucov)
                self.bcon = np.einsum("p...->...p", self.bcon)
                self.bcov = np.einsum("p...->...p", self.bcov)
        else:
            self.prims = np.ascontiguousarray(P)
            if calc_cons:
                self.cons = U
        # We no longer need the originals
        del P,G
        if calc_cons:
            del U

    # Act like a dict when retrieving lots of different things --
    # just compute/retrieve them on the fly!
    def __getitem__(self, key):
        if key in self.header['prim_names']:
            i = self.header['prim_names'].index(key)
            if self._zones_first:
                return self.prims[:, :, :, i]
            else:
                return self.prims[i]
        elif key in self.__dict__:
            return self.__dict__[key]
        elif key in ['r', 'th', 'phi']:
            return getattr(self.grid.coords, key)(self.grid.coord_all())
        elif key in ['x', 'y', 'z']:
            return getattr(self.grid.coords, 'cart_' + key)(self.grid.coord_all())
        elif key in ['X1', 'X2', 'X3']:
            return self.grid.coord_all()[int(key[-1:])]
        elif key in self.header:
            return self.header[key]
        elif key in vars.fns_dict:
            return vars.fns_dict[key](self)
        elif key[:4] == "log_":
            return np.log10(self[key[4:]])
        elif key[:3] == "ln_":
            return np.log(self[key[3:]])
        elif key[:4] == "pdf_":
            var_og = self[key[4:]]
            pdf_window=(np.min(var_og), np.max(var_og))
            return np.histogram(var_og, bins=100, range=pdf_window,
                weights=np.repeat(self.gdet, self.N3).reshape(var_og.shape), density=True)
        else:
            raise ValueError("IharmDump cannot find or compute {}".format(key))

    # TODO does __del__ need to do anything?



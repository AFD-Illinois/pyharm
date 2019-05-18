# Wrapper class for using iharm functions in C from Python programs

import os
import subprocess
import ctypes
import numpy as np

from defs import Loci


class Iharmc:
    _dll = None

    def __init__(self):
        # Find our own path, load the dll, which should be kept next to this file
        self._dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libiharm.so")
        if os.path.exists(self._dll_path):
            self._load_dll(self._dll_path)
        else:
            subprocess.call("make", cwd=os.path.dirname(self._dll_path))
            self._load_dll(self._dll_path)

    def _load_dll(self, path):
        # loads the library and registers function signatures
        self._dll = ctypes.CDLL(path)
        self._dll.U_to_P.argtypes = [ctypes.c_double, ctypes.c_double,
                                     np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     ctypes.c_double,
                                     np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ]
        self._dll.u2p.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  ctypes.c_double,
                                  np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                  np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    def U_to_P(self, params, G, U, P, Pout=None, iter_max=None):
        sh = G.shapes

        # Don't modify primitives in-place by default, and
        # use the opportunity to let numpy make them contiguous
        if Pout is None:
            Pout = np.ascontiguousarray(P.copy())

        pflag = np.ascontiguousarray(np.zeros(sh.grid_scalar, dtype=np.int32))

        self._dll.u2p(Pout,
                      np.ascontiguousarray(U),
                      np.ascontiguousarray(params['gam']),
                      pflag,
                      np.ascontiguousarray(G.gcov[Loci.CENT.value]),
                      np.ascontiguousarray(G.gcon[Loci.CENT.value]),
                      np.ascontiguousarray(G.gdet[Loci.CENT.value]),
                      np.ascontiguousarray(G.lapse[Loci.CENT.value]),
                      params['n_prims'], G.N[1], G.N[2], G.N[3], G.NG)

        return Pout, pflag

    def u2p(self, dump):
        """..."""
        # Keep a handle in case numpy copies the prims to make them contiguous
        prims_cont = np.ascontiguousarray(dump.prims)
        pflag = np.zeros((dump.N1, dump.N2, dump.N3), dtype=np.int32)

        self._dll.u2p(prims_cont,
                      np.ascontiguousarray(dump.cons),
                      np.ascontiguousarray(dump.header['gam']),
                      np.ascontiguousarray(pflag),
                      np.ascontiguousarray(dump.gcov),
                      np.ascontiguousarray(dump.gcon),
                      np.ascontiguousarray(dump.gdet),
                      np.ascontiguousarray(dump.lapse),
                      dump.prims.shape[0], dump.N1, dump.N2, dump.N3, 0)
        # Make sure we assign the prims at the end
        dump.prims = prims_cont


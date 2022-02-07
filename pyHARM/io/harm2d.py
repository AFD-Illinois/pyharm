# Functions for reading iharm2d_v3 dumps in ASCII format

import numpy as np

from .. import parameters

from .interface import DumpFile

class HARM2DFile(DumpFile):
    """
    Read iharm2d_v3 dumps.
    Caches all values on creation, which is bad for larger file types.
    """

    @classmethod
    def get_dump_time(cls, fname):
        data = np.loadtxt(fname, max_rows=1)
        return data[0]

    def __init__(self, fname):
        self.fname = fname
        # First line is header
        self.params = self.read_params()
        # Just cache the whole file, it's 2D
        self.data = np.loadtxt(fname, skiprows=1)
        self.sz = (self.params['n1'], self.params['n2'], self.params['n3'])
        self.prims = self.data[:,2:10].reshape(*self.sz, self.params['n_prim']).transpose(3,0,1,2)

    def read_params(self):
        params = {}

        data = np.loadtxt(self.fname, max_rows=1)

        # first get line 0 header info
        params['t']       = data[0]
        params['n1']      = int(data[1])
        params['n2']      = int(data[2])
        params['startx1'] = data[3]
        params['startx2'] = data[4]
        params['dx1']     = data[5]
        params['dx2']     = data[6]
        params['tf']   = data[7]
        params['n_step']  = int(data[8])
        params['gam']     = data[9]
        params['cour']    = data[10]
        params['dump_cadence']  = params['full_dump_cadence'] = data[11]
        params['log_cadence']   = data[12]
        params['image_cadence']     = data[13]
        params['restart_cadence']   = data[14]
        params['n_dump']    = int(data[15])
        params['image_cnt'] = int(data[16])
        params['rdump_cnt'] = int(data[17])
        params['dt']        = data[18]

        # Then add anything else we need.
        params['startx3'] = 0
        params['dx3'] = 0
        params['n3'] = 1
        params['n_prim'] = 8
        # TODO try to read these?
        params['coordinates'] = "mks"
        params['r_out'] = np.exp(params['startx1'] + params['n1']*params['dx1'])
        params['a'] = 0.9375
        params['hslope'] = 0.3
        # OR
        #params['coordinates'] = "cartesian"

        return parameters.fix(params)

    def read_var(self, var, slc=(), **kwargs):
        """Read the header and primitives from an iharm2d v3 ASCII file.
        No analysis or extra processing is performed
        @return P, params in standard pyHARM
        """
        i = self.index_of(var)
        if i is not None:
            return self.prims[i][slc]
        elif var == "ucon": # iharm2d caches u/b, we could fetch those
            return None
        elif var == "bcon":
            return None
        elif var == "jcon":
            return self.data[:,32:36].reshape(*self.sz, 4).transpose(3,0,1,2)[slc]
        elif var == "divB":
            return self.data[:,10].reshape(*self.sz)[slc]
        else:
            return None

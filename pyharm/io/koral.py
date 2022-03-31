# i/o for KORAL files

import h5py
import numpy as np

from .. import parameters
from .interface import DumpFile
from .iharm3d_header import read_hdr

class KORALFile(DumpFile):
    """File filter class for KORAL dump files.
    """

    @classmethod
    def get_dump_time(cls, fname):
        """Quickly get just the simulation time represented in the dump file.
        For cutting on time without loading everything
        """
        with h5py.File(fname, 'r') as dfile:
            if 't' in dfile.keys():
                return dfile.attrs['t']
            else:
                return None

    def __init__(self, filename, ghost_zones=False):
        """Create a KORALFile object. Note that ghost_zones does NOTHING for now,
        as no available KORAL files include ghost zones.
        """
        self.fname = filename
        self.file = h5py.File(filename, "r")
        self.params = self.read_params()
        self.params['ghost_zones'] = ghost_zones

    # def __del__(self):
    #     self.file.close()

    def read_params(self, **kwargs):
        # TODO this probably needs t, etc, etc unless KORAL puts those in the header
        params = read_hdr(self.file['/header'])
        return parameters.fix(params)

    def read_dump(self, var, **kwargs):
        """Read the header and primitives from a write_dump.
        No analysis or extra processing is performed
        @return P, params
        """
        for pair in (('RHO','rho'), ('UU','uint')):
            if var == pair[0]:
                var = pair[1]
        return self._prep_array(self.file['/quants/'+var][()], **kwargs)

    def _prep_array(arr, as_double=False, zones_first=False, add_ghosts=False):
        """Re-order and optionally up-convert an array from a file,
        to put it in usual pyharm order/format
        """
        # Upconvert to doubles if necessary
        # TODO could add other types?  Not really necessary yet
        if as_double:
            arr = arr.astype(np.float64)
        
        return arr

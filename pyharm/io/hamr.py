__license__ = """
 File: hamr.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020-2023, Ben Prather and AFD Group at UIUC
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

import os
import h5py
import numpy as np

from .. import parameters

from .interface import DumpFile

class HAMRFile(DumpFile):
    """File filter class for H-AMR dump files, specifically the "grmonty" regular-grid
    files in HDF5 format.  Usually used through file-agnostic interface:
    see file_reader and the FluidState class for details.
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
        """Create an HAMRFile object.  Note that ghost_zones does NOTHING for now,
        as no available HAMR files include ghost zones.
        """
        self.fname = filename
        self.file = h5py.File(filename, "r")
        self.params = self.read_params()
        self.params['ghost_zones'] = ghost_zones

    # def __del__(self):
    #     self.file.close()

    def read_params(self, **kwargs):
        params = {}

        # Per-write_dump single variables
        # dt? n_nstep? n _dump?
        # What about dscale?
        for hdr_key, par_key in [('t','t'), ('gam', 'gam'), ('dscale', 'dscale'),
                    ('a','a'), ('hslope', 'hslope'), ('N1','n1'), ('N2','n2'), ('N3','n3'),
                    ('R0','r0'), ('Rin','r_in'), ('Rout','r_out')]:
            if hdr_key in infile.attrs:
                params[par_key] = infile.attrs[hdr_key]
        
        params['dx1'], params['dx2'], params['dx3'] = infile.attrs['dx']

        # Translate header variables.  Taken from ipole.
        params['startx1'] = infile.attrs['startx'][0] - params['dx1']/2
        params['startx2'] = infile.attrs['startx'][1] - params['dx2']/2
        params['startx3'] = 0 # infile.attrs['startx'][2]

        # Translate from hamr x2 \in (-1, 1) -> mks x2 \in (0, 1)
        params['startx2'] = (params['startx2'] + 1)/2.
        #params['stopx2'] = (stopx[2] + 1)/2.
        params['dx2'] /= 2
        
        return parameters.fix(params)

    def read_var(self, var, **kwargs):
        """Read the header and primitives from a write_dump.
        No analysis or extra processing is performed
        @return P, params
        """
        return self._prep_array(self.file[var][()], **kwargs)

    def _prep_array(self, arr, astype=None):
        """Re-order and optionally up-convert an array from a file,
        to put it in usual pyharm order/format
        """
        # Ravel
        n1, n2, n3 = self.params['n1'], self.params['n2'], self.params['n3']
        arr = arr.reshape((n1, n2, n3))
        # Retype
        if astype is not None:
            arr = arr.astype(astype)

        return arr

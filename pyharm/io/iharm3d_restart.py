__license__ = """
 File: iharm3d_restart.py
 
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

import h5py
import numpy as np

from pyharm.io.iharm3d import Iharm3DFile

from .. import parameters
from .interface import DumpFile
from .iharm3d import Iharm3DFile
from .iharm3d_header import _write_value

__doc__ = \
"""Functions for reading and writing iharm3d-format restart files.
Since these files are quite simple, this is also the format used
by KHARMA when resizing and resuming simulations at higher resolution. 
"""

class Iharm3DRestart(Iharm3DFile):
    """File filter class for iharm3d restart files. Overrides just the parameters & read_var methods.
    """

    def read_params(self, **kwargs):
        """Read the file header and per-dump parameters (t, dt, etc)"""
        with h5py.File(self.fname, "r") as fil:
            params = {}
            # Add everything a restart file records
            for key in ['DTd', 'DTf', 'DTl', 'DTp', 'DTr', 'cour', 'dt', 'dump_cnt',
                        'gam', 'n1', 'n2', 'n3', 'nstep', 'restart_id', 't', 'tdump',
                        'tf', 'tlog', 'version']:
                if key in fil:
                    params[key] = fil[key][()]
            if 'x1Min' in fil.keys():
                for key in ['x1Min', 'x1Max', 'x2Min', 'x2Max', 'x3Min', 'x3Max']:
                    if key in fil:
                        params[key.lower()] = fil[key][()]
            if 'a' in fil.keys():
                for key in ['a', 'hslope', 'R0', 'Rhor', 'Rin', 'Rout']:
                    if key in fil:
                        params[key] = fil[key][()]
                params['coordinates'] = 'fmks'
            else:
                params['coordinates'] = 'cartesian'

            # Since restarts don't list names, always fall back to convention
            self.prim_names = None

            return parameters.fix(params)

    def read_var(self, var, slc=(), **kwargs):
        if var in self.cache:
            return self.cache[var]
        with h5py.File(self.fname, "r") as fil:
            # Translate the slice to a portion of the file
            # TODO ghost zones
            fil_slc = [slice(None), slice(None), slice(None)]
            if isinstance(slc, tuple) or isinstance(slc, list):
                for i in range(len(slc)):
                    if isinstance(slc[i], int) or isinstance(slc[i], np.int32) or isinstance(slc[i], np.int64):
                        fil_slc[2-i] = slice(slc[i], slc[i]+1)
                    else:
                        fil_slc[2-i] = slc[i]
            fil_slc = tuple(fil_slc)

            # No indications present in restarts to read any fancy indexing. Only support the basics
            i = self.index_of(var)
            if i is not None:
                #print("Reading file slice ", (i,) + fil_slc)
                arr = fil['/p'][(i,) + fil_slc]
                if len(arr.shape) > 3:
                    self.cache[var] = arr.transpose(0,3,2,1)
                else:
                    self.cache[var] = arr.transpose(2,1,0)
                return self.cache[var]
            else:
                raise IOError("Cannot find variable "+var+" in file "+self.fname+"!")

## Module functions

def write_restart(dump, fname, astype=np.float64):
    """Write a valid iharm3d restart/KHARMA resize file,
    containing the data in FluidState 'dump', to 'fname', at precision 'astype'.
    """
    with h5py.File(fname, "w") as outf:

        # Record this was converted
        _write_value(outf, "pyharm-converter-0.1", 'version')
        # Variables needed for restarting
        outf['n1'] = dump['n1']
        outf['n2'] = dump['n2']
        outf['n3'] = dump['n3']
        outf['gam'] = dump['gam']
        outf['cour'] = dump['cour']
        outf['t'] = dump['t']
        outf['dt'] = dump['dt']
        # Always write native coordinate bounds exactly.
        # iharm3d reconstitutes these in MKS, but
        # KHARMA needs exact values
        outf['x1Min'] = dump['x1min']
        outf['x1Max'] = dump['x1max']
        outf['x2Min'] = dump['x2min']
        outf['x2Max'] = dump['x2max']
        outf['x3Min'] = dump['x3min']
        outf['x3Max'] = dump['x3max']
        if 'tf' in dump.params:
            outf['tf'] = dump['tf']
        elif 'tlim' in dump.params:
            outf['tf'] = dump['tlim']
        if 'a' in dump.params:
            outf['a'] = dump['a']
            outf['hslope'] = dump['hslope']
            outf['Rhor'] = dump['r_eh']
            outf['Rin'] = dump['r_in']
            outf['Rout'] = dump['r_out']
            outf['R0'] = 0.0
        if 'n_step' in dump.params:
            outf['nstep'] = dump['n_step']
        if 'n_dump' in dump.params:
            outf['dump_cnt'] = dump['n_dump']
        if 'game' in dump.params:
            outf['game'] = dump['game']
            outf['gamp'] = dump['gamp']
            # This one seems unnecessary?
            outf['fel0'] = dump['fel0']

        # Every KHARMA dump is full
        outf['DTd'] = dump['dump_cadence']
        outf['DTf'] = dump['dump_cadence']
        # These aren't recorded from KHARMA
        outf['DTl'] = 0.1
        outf['DTp'] = 100
        outf['DTr'] = 10000
        # I dunno what this is
        outf['restart_id'] = 100
        
        if 'next_dump_time' in dump.params:
            outf['tdump'] = dump['next_dump_time']
        else:
            outf['tdump'] = dump['t'] + dump['dump_cadence']
        if 'next_log_time' in dump.params:
            outf['tlog'] = dump['next_log_time']
        else:
            outf['tlog'] = dump['t'] + 0.1

        # This will fetch and write all primitive variables,
        # sans ghost zones as is customary for iharm3d restart files
        G = dump.grid
        if G.NG > 0:
            p = dump.reader.read_var('prims', astype=astype)
            outf["p"] = np.einsum("pijk->pkji", p[G.slices.allv + G.slices.bulk]).astype(astype)
        else:
            p = dump.reader.read_var('prims', astype=astype)
            outf["p"] = np.einsum("pijk->pkji", p).astype(astype)
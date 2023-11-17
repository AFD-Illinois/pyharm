__license__ = """
 File: iharm3d.py
 
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

from .interface import DumpFile

# Treat header i/o as a part of this file,
# but don't balloon the line count
from .iharm3d_header import read_hdr, write_hdr, _write_value

__doc__ = \
"""Functions for reading and writing iharm3d/Illinois HDF format files,
as well as iharm3d's log files.
"""

class Iharm3DFile(DumpFile):
    """File filter class for Illinois HDF5 format dump files, from iharm3d, ebhlight, pyharm, or converted
    from KHARMA output or potentially more exotic formats.
    Usually used through file-agnostic interface: see file_reader and the FluidState class for details.
    """

    @classmethod
    def get_dump_time(cls, fname):
        """Quickly get just the simulation time represented in the dump file.
        For cutting on time without loading everything.
        """
        with h5py.File(fname, 'r') as dfile:
            if 't' in dfile.keys():
                return dfile['t'][()]
            else:
                return None

    def __init__(self, filename, ghost_zones=False, params=None):
        """Create an Iharm3DFile object -- note that the file handle will stay
        open as long as the object.
        """
        self.fname = filename
        self.cache = {}
        if params is None:
            self.params = self.read_params()
            #self.params['ghost_zones'] = ghost_zones
            #self.params['ng_file'] = self.params['ng']
            #self.params['ng'] = ghost_zones * self.params['ng']
        else:
            self.params = params

    def __del__(self):
        # Try to clean up what we can. Anything that may possibly not be a simple ref
        for cache in ('cache', 'params'):
            if cache in self.__dict__:
                del self.__dict__[cache]

    def read_params(self, **kwargs):
        """Read the file header and per-dump parameters (t, dt, etc)"""
        with h5py.File(self.fname, "r") as fil:
            params = read_hdr(fil['/header'])

            # Add variables which change per-dump, recorded outside header
            for key in ['t', 'dt', 'n_step', 'n_dump', 'is_full_dump', 'dump_cadence', 'full_dump_cadence']:
                if key in fil and not isinstance(fil[key], h5py.Group):
                    params[key] = fil[key][()]

            # Grab the git revision if it's available, as this isn't recorded to/read from the header either
            if 'extras' in fil and 'git_version' in fil['extras']:
                params['git_version'] = fil['/extras/git_version'][()].decode('UTF-8')

            # Unlike in most codes (including kharma!) iharm3d is self-documenting about every primitive
            # it contains.  Use that to advantage here, circumventing the need to guess.
            params['prim_names']  = [prim_name.decode() for prim_name in fil['header/prim_names']]

            return params

    def index_of(self, var):
        # Add any prim names we get from the file, but fall back to guessing from the usual ordering
        prim_names = self.params['prim_names']
        if (prim_names is not None) and (var in prim_names):
            return prim_names.index(var.upper())
        else:
            return DumpFile.index_of(var)
        

    def read_var(self, var, slc=(), **kwargs):
        if var in self.cache:
            return self.cache[var]
        with h5py.File(self.fname, "r") as fil:
            # Translate the slice to a portion of the file
            # A bit overkill to stay adaptable
            # TODO ghost zones
            fil_slc = [slice(None), slice(None), slice(None)]
            if isinstance(slc, tuple) or isinstance(slc, list):
                for i in range(len(slc)):
                    if isinstance(slc[i], int) or isinstance(slc[i], np.int32) or isinstance(slc[i], np.int64):
                        fil_slc[i] = slice(slc[i], slc[i]+1)
                    else:
                        fil_slc[i] = slc[i]
            fil_slc = tuple(fil_slc)

            i = self.index_of(var)
            if i is not None:
                # This is one of the main vars in the 'prims' array
                self.cache[var] = self._prep_array(fil['/prims'][fil_slc + (i,)], **kwargs)
                return self.cache[var]
            else:
                # This is something else we should grab by name
                # Default to int type for flags
                if "flag" in var and 'astype' not in kwargs:
                    kwargs['astype'] = np.int32
                # Read desired slice
                if var in fil:
                    self.cache[var] = self._prep_array(fil[var][fil_slc], **kwargs)
                    return self.cache[var]
                elif var in fil['/extras']:
                    self.cache[var] = self._prep_array(fil['/extras/'+var][fil_slc], **kwargs)
                    return self.cache[var]
                else:
                    raise IOError("Cannot find variable "+var+" in file "+self.fname+"!")

    def _prep_array(self, arr, astype=None):
        """Re-order and optionally up-convert an array from a file,
        to put it in usual pyharm order/format
        """
        # Reverse indices on vectors, since most pyharm tooling expects p,i,j,k
        # See iharm_dump for analysis interface that restores i,j,k,p order
        if len(arr.shape) > 3:
            arr = np.einsum("...m->m...", arr)

        # Convert to desired type. Useful for flags.
        if astype is not None:
            arr = arr.astype(astype)
        
        return arr

## Module functions

def read_log(logfname):
    """Read an iharm3d-format log.out file into a dictionary
    """
    dfile = np.loadtxt(logfname).transpose()
    # iharm3d's logs are deep magic. TODO header?
    log = {}
    log['t'] = dfile[0]
    log['rmed'] = dfile[1]
    log['pp'] = dfile[2]
    log['e'] = dfile[3]
    log['uu_rho_gam_cent'] = dfile[4]
    log['uu_cent'] = dfile[5]
    log['mdot'] = dfile[6]
    log['edot'] = dfile[7]
    log['ldot'] = dfile[8]
    log['mass'] = dfile[9]
    log['egas'] = dfile[10]
    log['Phi'] = dfile[11]
    log['phi'] = dfile[12]
    log['jet_EM_flux'] = dfile[13]
    log['divbmax'] = dfile[14]
    log['lum_eht'] = dfile[15]
    log['mdot_eh'] = dfile[16]
    log['edot_eh'] = dfile[17]
    log['ldot_eh'] = dfile[18]

    return log

def write_dump(dump, fname, astype=np.float32, ghost_zones=False):
    """Write the data in FluidState 'dump' to file 'fname' in iharm3d/Illinois HDF format.
    """
    with h5py.File(fname, "w") as outf:


        # Fill in a gap we can only do here
        if 'n_prim' not in dump.params:
            dump.params['n_prim'] = dump['prims'].shape[0]

        write_hdr(dump.params, outf)

        # Per-dump single variables
        if 't' in dump.params:
            outf['t'] = dump['t']
        else:
            outf['t'] = 0
        if 'dt' in dump.params:
            outf['dt'] = dump['dt']
        else:
            outf['dt'] = 0.1
        if 'dump_cadence' in dump.params:
            outf['dump_cadence'] = dump['dump_cadence']
            outf['full_dump_cadence'] = dump['dump_cadence']
        else:
            outf['dump_cadence'] = 5
            outf['full_dump_cadence'] = 5
        outf['is_full_dump'] = True
        if 'n_dump' in dump.params:
            outf['n_dump'] = dump['n_dump']
        if 'n_step' in dump.params:
            outf['n_step'] = dump['n_step']

        # This will fetch and write all primitive variables
        G = dump.grid
        if G.NG > 0 and not ghost_zones:
            p = dump['prims'].astype(astype)
            outf["prims"] = np.einsum("p...->...p", p[G.slices.allv + G.slices.bulk]).astype(astype)
        else:
            p = dump['prims'].astype(astype)
            outf["prims"] = np.einsum("p...->...p", p).astype(astype)

        # Extra in-situ calculations or custom debugging additions
        if "extras" not in outf:
            outf.create_group("extras")
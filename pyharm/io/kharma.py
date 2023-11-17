__license__ = """
 File: kharma.py
 
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

import sys
import glob
import numpy as np
import pandas
import h5py

from .. import parameters
from ..util import slice_to_index, i_of
from ..defs import Loci
from ..grid import Grid
from .interface import DumpFile

# This is where we drop in the Parthenon reader
from .phdf import phdf
from .phdf_old import phdf as phdf_old

__doc__ = \
"""Read KHARMA output files and logs.  Pretty much supports any Parthenon code including (interpolated) AMR.
Contains much index math.
"""

class KHARMAFile(DumpFile):
    """File filter for KHARMA files"""
    # Names which aren't directly prims.x or cons.x, but which we can translate
    prim_names_dict = {"RHO": "rho",
                       "UU": "u",
                       "U1": "u1",
                       "U2": "u2",
                       "U3": "u3",
                       "KTOT": "Ktot",
                       "KEL_KAWAZURA": "Kel_Kawazura",
                       "KEL_WERNER":   "Kel_Werner",
                       "KEL_ROWAN":    "Kel_Rowan",
                       "KEL_SHARMA":   "Kel_Sharma",
                       "KEL_CONSTANT": "Kel_Constant"}
    # The ordering of primitive variables, for returning "prims" and "cons"
    # The file doesn't need to contain all these, they just need to be in
    # the order they would appear
    # If iharm3d ever supports viscous AND e- together we're out of spec here & in general
    var_names_ordered = ['rho', 'u', 'u1', 'u2', 'u3', 'B1', 'B2', 'B3', 'q', 'dP',
                         'Ktot', 'Kel_Constant', 'Kel_Werner', 'Kel_Rowan', 'Kel_Sharma']


    @classmethod
    def get_dump_time(cls, fname):
        """Quickly get just the simulation time represented in the dump file.
        For cutting on time without loading everything
        """
        with h5py.File(fname, 'r') as dfile:
            if 'Info' in dfile.keys():
                return dfile['Info'].attrs['Time']
        return None

    def kharma_standardize(self, var):
        """Standardize the names we're asked for, so we cache only one copy"""
        # Translate certain caps
        if var in self.prim_names_dict.keys():
            var = self.prim_names_dict[var]

        # Pick out indices and return their vectors
        ind = None
        if var[-1:] in ("1", "2", "3"):
            # Mark which index we want
            ind = int(var[-1:]) - 1
            # Then read the corresponding vector, cons/prims.u/B
            var = var[:-2] + ("B" if "B" in var[-2:] else "uvec")

        # Extend the shorthand for primitive variables to their full names in KHARMA,
        # but not other variables.
        if var in ("rho", "u", "uvec", "B", "q", "dP", "psi"):
            var = "prims."+var
        if ("Kel" in var or "Ktot" in var) and ("cons" not in var):
            var = "prims."+var

        return var, ind

    def __init__(self, filename, ghost_zones=False, params=None):
        """Create an Iharm3DFile object -- note that the file handle will stay
        open as long as the object
        """
        self.fname = filename
        self.cache = {}
        if params is None:
            self.params = self.read_params()
            # Adjust some params based on args to __init__
            self.params['ghost_zones'] = ghost_zones
            self.params['ng'] = ghost_zones * self.params['ng_file']
            if ghost_zones and self.params['ng_file'] == 0:
                raise ValueError("Ghost zones aren't available in file {}".format(self.fname))
        else:
            self.params = params

    def __del__(self):
        # Try to clean up what we can. Anything that may possibly not be a simple ref
        for cache in ('cache', 'params'):
            if cache in self.__dict__:
                del self.__dict__[cache]

    def read_params(self):
        # TODO there's likely a nice way to get all this from phdf
        try:
            fil = phdf(self.fname)
        except:
            fil = phdf_old(self.fname)

        params = None
        if 'Input' in fil.fid:
            par_string = fil.fid['Input'].attrs['File']
            if type(par_string) != str:
                par_string = par_string.decode("UTF-8")
            params = parameters.parse_parthenon_dat(par_string)
        else:
            # Read from the closest parameter file
            path1 = "/".join(self.fname.split("/")[:-1])+"/*.par"
            fnames = glob.glob(path1)
            if len(fnames) == 0:
                path2 = "/".join(self.fname.split("/")[:-2])+"/*.par"
                fnames = glob.glob(path2)
            if len(fnames) == 0:
                path3 = "/".join(self.fname.split("/")[:-1])+"*.par"
                fnames = glob.glob(path3)
            if len(fnames) == 0:
                raise IOError("Cannot read parameter file KHARMAv1 dump {}.\nMove associated .par file to the dump's directory.".format(self.fname))

            with open(fnames[-1], 'r') as parfile:
                params = parameters.parse_parthenon_dat(parfile.read())

        if params is None:
            raise RuntimeError("No parameters could be found in KHARMA dump {}".format(self.fname))

        # Use Parthenon's reader for the file-specific stuff
        params['ng_file'] = fil.NGhost * fil.IncludesGhost
        # Set incidental parameters from what we've read
        params['t'] = fil.Time
        params['n_step'] = fil.NCycle
        params['num_blocks'] = fil.NumBlocks
        # Add dump number if we've conformed to usual naming scheme
        fname_parts = self.fname.split("/")[-1].split(".")
        if len(fname_parts) > 2:
            try:
                params['n_dump'] = int(fname_parts[2])
            except ValueError:
                # dumps named 'final'.  Drop them?
                params['n_dump'] = fname_parts[2]
        
        # Regenerating this takes *forever*
        # Cache it in 'params' so it follows the file around *everywhere*
        params['phdf_aux'] = {}
        params['phdf_aux']['offset'] = fil.offset
        params['phdf_aux']['isGhost'] = fil.isGhost
        params['phdf_aux']['BlockIdx'] = fil.BlockIdx
        params['phdf_aux']['BlockBounds'] = fil.BlockBounds

        fil.fid.close()
        del fil
        return params

    def read_var(self, var, astype=None, slc=(), out=None, skip_cache=False, fail_if_not_found=True):
        """Read a variable from the backing file.  Opens, reads, closes.
        Pass the 'out' parameter at your own risk, it must be the right size to match 'slc'
        """
        #print("Reading",var)
        var, ind = self.kharma_standardize(var)
        if var in self.cache:
            if ind is not None:
                return self.cache[var][ind]
            else:
                return self.cache[var]

        # Open file
        try:
            fil = phdf(self.fname, **self.params['phdf_aux'])
        except:
            fil = phdf_old(self.fname)

        # All primitive or conserved vars. Generally used
        # for converting file formats.
        if var == "prims" or var == "cons":
            # Reshape rho to 4D by adding a rank in front for prim index
            # TODO THIS DOES NOT WORK FOR KHARMA HARM-DRIVER RESTARTS
            kwargs = {'astype': astype, 'slc': slc, 'fail_if_not_found': fail_if_not_found}
            all_vars = self.read_var(var+'.rho', **kwargs)[np.newaxis, Ellipsis]
            for v2 in self.var_names_ordered[1:]:
                try:
                    new_var = self.read_var(v2, **kwargs)
                    if len(new_var.shape) < len(all_vars.shape):
                        # Reshape to 4D if needed to append
                        new_var = new_var[np.newaxis, Ellipsis]
                    all_vars = np.append(all_vars, new_var, axis=0)
                except (IOError, OSError, TypeError, IndexError, KeyError):
                    # Not every file will have all prims
                    pass
            # Save the result
            self.cache[var] = all_vars
            return self.cache[var]

        # Don't require this prefix reading from old files
        if "c.c.bulk."+var in fil.Variables:
            var = "c.c.bulk."+var

        if var not in fil.Variables and self.index_of(var) is None:
            # Try indexing/fetching by name without the prefix
            if self.index_of(var.replace("prims.", "")) is not None:
                var = var.replace("prims.", "")
        # Try to get prims.B from cons.B (for e.g. KHARMA restarts)
        # Note B1,2,3->B,ind already so we have to reform cons.B1,2,3 (should take as arg)
        # We don't try this with other variables, one could maybe?
        elif var in ["B", "prims.B"] and "prims.B" not in fil.Variables and "cons.B" in fil.Variables:
            grid = Grid(self.params)
            var_con = 'cons.B'+str(ind+1) if ind is not None else 'cons.B'
            return self.read_var(var_con, astype=astype, slc=slc) / \
                    grid['gdet'][grid.slices.geom_slc(slc)]

        params = self.params
        # Recall ng=0 if ghost_zones is False.  Thus this says:
        # if we want ghost zones, set them in nontrivial dimensions
        ng_ix = params['ng']
        ng_iy = params['ng'] if params['n2'] > 1 else 0
        ng_iz = params['ng'] if params['n3'] > 1 else 0
        ntot = [params['n1']+2*ng_ix, params['n2']+2*ng_iy, params['n3']+2*ng_iz]
        # Even if we don't want ghosts, we'll potentially need to cut them from the file
        ngf = [params['ng_file'],
               params['ng_file'] if params['n2'] > 1 else 0,
               params['ng_file'] if params['n3'] > 1 else 0]
        # Finally, we need to decipher where to put each meshblock vs the whole grid,
        # which we do inelegantly using zone locations
        dx = (params['dx1'], params['dx2'], params['dx3'])
        startx = (params['startx1'], params['startx2'], params['startx3'])

        # What locations do we want to read from the file?
        # Get a start and end point based on the *total* size and slice
        file_start, file_stop = slice_to_index((0, 0, 0), ntot, slc)
        out_shape = [file_stop[i] - file_start[i] for i in range(len(file_stop))]
        # Now that we know which dimensions *stay* non-trivial, revise our output size slicing
        ng = [params['ng'] if out_shape[0] > 1 else 0,
              params['ng'] if out_shape[1] > 1 else 0,
              params['ng'] if out_shape[2] > 1 else 0]

        #print("Reading slice", slc, " of file, indices ", file_start, " to ", file_stop, " to shape ", out_shape)

        if out is None:
            # Allocate the full output mesh size
            if "jcon" in var:
                out = np.zeros((4, *out_shape), dtype=astype)
            elif var.split(".")[-1][:1] == "B" or var.split(".")[-1] == "uvec": # We cache the whole thing even for an index
                out = np.zeros((3, *out_shape), dtype=astype)
            else:
                out = np.zeros(out_shape, dtype=astype)

        # Arrange and read each block
        for ib in range(fil.NumBlocks):
            bb = fil.BlockBounds[ib]
            # How much smaller is this block's dx vs the file norm?
            block_dx = np.abs(bb[1] - bb[0])/fil.MeshBlockSize[0]
            level = int(round(dx[0] / block_dx))
            #print("Reading block level", level)
            # Internal location of the block i.e. starting/stopping physical indices in the final, big mesh
            # First, take the start/stop locations and map them to integers
            # We only need to add ghost zones here if the file has them *and* we want them:
            # If so, each block will be 2*ng bigger, but we'll want to handle indices from zero regardless
            b = tuple([slice(int((bb[2*i]   + dx[i]/2 - startx[i])/dx[i]) + ng[i],
                             int((bb[2*i+1] + dx[i]/2 - startx[i])/dx[i]) + ng[i]) for i in range(3)])
            # Intersect block's global bounds with our desired slice: this is where we're outputting to,
            # on the (never instantiated) global grid
            loc_slc = tuple([slice(max(b[i].start, file_start[i]), min(b[i].stop, file_stop[i])) for i in range(3)])
            # Subtract off the start of the slice: this is where we're outputting to in our real array
            out_slc = tuple([slice(loc_slc[i].start - file_start[i] - ng[i]//level, loc_slc[i].stop - file_start[i] + ng[i]//level) for i in range(3)])
            # Subtract off the block's global starting point: this is what we're taking from in the block
            # If the ghost zones are included (ng_f > 0) but we don't want them (all) (ng_i = 0),
            # then take a portion of the file.  Otherwise take it all.
            fil_slc = (slice((loc_slc[2].start - b[2].start)*level + ngf[2] - ng[2], (loc_slc[2].stop - b[2].start)*level + ngf[2] + ng[2], level),
                       slice((loc_slc[1].start - b[1].start)*level + ngf[1] - ng[1], (loc_slc[1].stop - b[1].start)*level + ngf[1] + ng[1], level),
                       slice((loc_slc[0].start - b[0].start)*level + ngf[0] - ng[0], (loc_slc[0].stop - b[0].start)*level + ngf[0] + ng[0], level))
            if (fil_slc[-3].start > fil_slc[-3].stop) or (fil_slc[-2].start > fil_slc[-2].stop) or (fil_slc[-1].start > fil_slc[-1].stop):
                # Don't read blocks outside our domain
                #print("Skipping block: ", b, " would be to location ", out_slc, " from portion ", fil_slc)
                continue
            #print("Reading var ", var, " from block: ", b, " to location ", out_slc, " by reading block portion ", fil_slc)

            if 'prims.rho' in fil.Variables:
                if var not in fil.fid:
                    raise IOError("Cannot read variable "+var+" from file "+self.fname+"!")
                # New file format. Read whatever
                if len(out.shape) == 4: # Always read the whole vector, even if we're returning an index
                    #print("Reading vector size ", fil.fid[var][(ib, slice(None)) + fil_slc].transpose(0,3,2,1).shape, " to loc size ", out[(slice(None),) + out_slc].shape)
                    try:
                        # Newer format: block, var, k, j, i on disk
                        out[(slice(None),) + out_slc] = fil.fid[var][(ib, slice(None)) + fil_slc].transpose(0,3,2,1)
                    except (IndexError, ValueError):
                        # Older format: block, k, j, i, var
                        out[(slice(None),) + out_slc] = fil.fid[var][(ib,) + fil_slc + (slice(None),)].T
                else: # Read a scalar, knocking off the extra index if necessary
                    #print("Reading scalar ", var, " on-disk size ", fil.fid[var].shape, " to loc size ", out[out_slc].shape)
                    #print("Using slice ", fil_slc)
                    try:
                        # Newest (and ironically, also oldest) format: scalars as k, j, i only
                        out[out_slc] = fil.fid[var][(ib,) + fil_slc].T
                    except (IndexError, ValueError):
                        try:
                            # Newer format: block, var, k, j, i on disk
                            out[out_slc] = fil.fid[var][(ib, 0) + fil_slc].T
                        except (IndexError, ValueError):
                            # Older format: block, k, j, i, var
                            out[out_slc] = fil.fid[var][(ib,) + fil_slc + (0,)].T

            else:
                # Old file formats.  First anything scalar:
                if var in fil.Variables:
                    out[(Ellipsis,) + out_slc] = fil.fid[var][(ib,) + fil_slc + (slice(None),)].T
                # If we'd split out "B" it was called "B_prim" (wasn't find/replaced above)
                elif var[0] == "B" and 'c.c.bulk.B_prim' in fil.Variables:
                        out[(slice(None),) + out_slc] = fil.fid['c.c.bulk.B_prim'][(ib,) + fil_slc + (slice(None),)].T
                else:
                    i = self.index_of(var)
                    if i is None:
                        # We're not grabbing anything except primitives from old KHARMA files.
                        raise IOError("Cannot read variable "+var+" from file "+self.fname+"!")
                    else:
                        # Both the int & slice cases require the same line: first 3 indices of file -> last 3 indices of output
                        #print("Read {} at {}".format(var, i))
                        #print(fil_slc + (i,), file=sys.stderr)
                        #print((Ellipsis,) + out_slc, file=sys.stderr)
                        out[(Ellipsis,) + out_slc] = fil.fid['c.c.bulk.prims'][(ib,) + fil_slc + (i,)].T
        # Close
        fil.fid.close()
        del fil

        # ALWAYS keep 3 indices.  Better to keep than to squeeze and accidentally broadcast
        if skip_cache:
            return out
        else:
            self.cache[var] = out
            if ind is not None:
                return self.cache[var][ind]
            else:
                return self.cache[var]

## Module functions

def read_log(fname):
    with open(fname) as inf:
        inf.readline()
        header = [e.split('=')[1].rstrip() for e in inf.readline().split('[')[1:]]

    tab = pandas.read_table(fname, delim_whitespace=True, comment='#', names=header)
    out = {}
    for name in header:
        out[name] = np.array(tab[name])

    if not 'time' in out:
        print("Not loading KHARMA log file: header not present!")
        return None
    
    # Files can contain multiple runs and restarts
    # First, start at the most recent zero (argmin returns all in order)
    start = len(out['time']) - np.argmin(out['time'][::-1]) - 1
    for name in header:
        out[name] = out[name][start:]

    # Then run forward, look for jumps back, and take the more recent run
    # The indices will all shift, so repeat from zero as necessary
    caught_up = False
    while not caught_up:
        t_last = -1
        caught_up = True
        for i,t in enumerate(out['time']):
            # This heuristic asks "did we jump back, and can we more or less refill the gap?"
            if t < t_last and i_of(out['time'][i:], t_last) > i - i_of(out['time'], t) - 100:
                # i_of returns only first occurrence
                i_start = i_of(out['time'], t)
                # i_of returns the index *before* current time
                i_end = i
                for name in header:
                    out[name] = np.append(out[name][:i_start], out[name][i_end:])
                caught_up = False
                break
            t_last = t

    return out

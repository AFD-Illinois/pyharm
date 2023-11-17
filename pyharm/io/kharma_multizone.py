__license__ = """
 File: kharma_multizone.py
 
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
import glob
import numpy as np

from .. import parameters
from ..util import slice_to_index, i_of
from ..defs import Loci
from ..grid import Grid
from .interface import DumpFile

# MZ file components are KHARMA files
from .kharma import KHARMAFile

__doc__ = \
"""Read KHARMA output files and logs.  Pretty much supports any Parthenon code including (interpolated) AMR.
Contains much index math.
"""

def mz_data_dir(n):
    """Data directory naming scheme"""
    return "{:05d}".format(n)

def get_mz_nums(run_num, nzones):
    n = run_num
    m = nzones - 1 # we don't repeat sims at the inner/outer edge
    d = n % m
    # This ordering ensures we overplot with the right data
    # From the edge run before last, up to the run replaced by ours
    old_part = list(range((n//m - 1)*m, n - 2*d))
    # From the last edge run up to this run, not including us
    # (The particular dump being plotted will represent this run)
    new_part = list(range((n//m)*m, n))

    return old_part + new_part


class KHARMAMZFile(DumpFile):
    """File filter for KHARMA multizone runs, which turns the appropriate files into a single FluidState"""

    @classmethod
    def get_dump_time(cls, fname):
        """Quickly get just the simulation time represented in the dump file.
        For cutting on time without loading everything
        """
        return KHARMAFile.get_dump_time(fname)

    def __init__(self, filename):
        """Create a KHARMAMZFile object -- note that the file handle will stay
        open as long as the object.
        Note you can't pass a parameters object in here, we modify a KHARMA params
        file and you can't out-meta that.
        """
        self.fname = filename
        self.cache = {}
        # Temporary file for reading/modifying parameters
        kfile_tmp = KHARMAFile(self.fname)
        # Record some basics about us
        # Make iteration count reflect paths and easier numbering
        self.iteration = int(kfile_tmp.params['config']['resize_restart']['iteration'])
        self.nzones = int(kfile_tmp.params['config']['resize_restart']['nzone'])
        self.base = int(kfile_tmp.params['config']['resize_restart']['base'])

        try:
            self.run_num = int(os.path.realpath(self.fname).split('/')[-2])
        except:
            print(os.path.realpath(self.fname), os.path.realpath(self.fname).split('/')[-2])
            raise RuntimeError("KHARMAMZFile must be in a multizone directory structure!")

        mz_nums = get_mz_nums(self.run_num, self.nzones)
        my_dir = '/'.join(self.fname.split('/')[:-1])
        # When imaging the first run, use beginning states of subsequent runs
        mz_names = []
        for num in mz_nums:
            if num < 0:
                mz_names.append(glob.glob(my_dir+'/../'+mz_data_dir(-num)+'/*.out0.00000.phdf')[0])
            else:
                mz_names.append(glob.glob(my_dir+'/../'+mz_data_dir(num)+'/*.out0.final.phdf')[0])
        mz_names.append(self.fname)
        # Add all the correct files, in order & with matching parameters
        self.kfiles = []
        for fname in mz_names:
            # Modify each file's params rather than replace
            # KHARMAFile stores some very specific stuff in there
            # TODO I have too many names for things
            self.kfiles.append(KHARMAFile(fname))
            self.kfiles[-1].params['n1tot'] = self.kfiles[-1].params['n1'] = kfile_tmp.params['n1']//2 * (self.nzones+1)
            self.kfiles[-1].params['r_in'] = self.base**0
            self.kfiles[-1].params['startx1'] = self.kfiles[-1].params['x1min'] = np.log(self.base**0)
            self.kfiles[-1].params['r_out'] = self.base**(self.nzones+1)
            self.kfiles[-1].params['stopx1'] = self.kfiles[-1].params['x1max'] = np.log(self.base**(self.nzones+1))


        # Take the current file's doctored params as our own,
        # add a couple useful things
        self.params = self.kfiles[-1].params
        self.params['r_in_active'] = kfile_tmp.params['r_in']
        self.params['r_out_active'] = kfile_tmp.params['r_out']
        self.params['startx1_active'] = kfile_tmp.params['startx1']
        self.params['stopx1_active'] = kfile_tmp.params['startx1'] + kfile_tmp.params['n1']*kfile_tmp.params['dx1']

        del kfile_tmp

    def __del__(self):
        # Try to clean up what we can. Anything that may possibly not be a simple ref
        for cache in ('params'):
            if cache in self.__dict__:
                del self.__dict__[cache]

    def read_var(self, var, astype=None, slc=(), fail_if_not_found=True):

        # We have to handle indices here since we're skipping the KHARMAFile cache/size handling
        ind = None
        if var[-1:] in ("1", "2", "3"):
            # Mark which index we want
            ind = int(var[-1:]) - 1

        # Create the full output array by calling the first read
        out = self.kfiles[0].read_var(var, astype=astype, slc=slc, skip_cache=True, fail_if_not_found=fail_if_not_found)
        # Add to the full output with the rest of the backing files
        for fil in self.kfiles[1:]:
            fil.read_var(var, astype=astype, slc=slc, out=out, skip_cache=True, fail_if_not_found=fail_if_not_found)

        # TODO caches here

        # Note to self: stop interpreting 'None' as 'cannot find' and start using 'throw' instead
        if ind is not None:
            return out[ind]
        else:
            return out

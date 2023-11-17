__license__ = """
 File: analysis.py
 
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
import numpy as np

from ..fluid_state import FluidState
from . import analyses

__doc__ = \
"""A bit like frame.py, this file exists mostly to offload code from the pyharm-analysis script.
"""

def write_ana_dict(out, out_full, n, n_dumps):
    """Write output of analyze() to a single HDF5 file.
    Note this is not thread-safe and must be called from one process
    """
    if out is None:
        print("Failed to read dump number {}".format(n))
        return
    for key in list(out.keys()):
        tag = key.split('/')[0]
        if key not in out_full:
            # Add destination ndarray of the right size if not present
            # Use single-precision, because we have rtht profiles that are entire movies!
            if tag == 't' or key == 'coord/t':
                out_full[key] = np.zeros(n_dumps, dtype=np.float32)
            elif tag[-1:] == 't':
                out_full[key] = np.zeros((n_dumps,)+out[key].shape, dtype=np.float32)
            else:
                out_full[key] = np.zeros_like(out[key])

        # Slot in time-dependent vars, add averaged vars to running total
        try:
            if tag[-1:] == 't' or key == 'coord/t':
                out_full[key][n] = out[key]
            else:
                out_full[key][()] += out[key]
        except TypeError as e:
            print("Encountered error when updating {}: {}".format(key, e))

def analyze(args):
    fname, kwargs = args
    out = {}
    dump = FluidState(fname)
    ana_types = kwargs['ana_types'].split(",")
    # Always start with "basic" as it's got some things we'll need
    if ana_types[0] != "basic":
        ana_types.insert(0, "basic")
    for type in ana_types:
        analyses.__dict__[type](dump, out, **kwargs)
    return out

def analyze_catch_err(args):
    try:
        return analyze(args)
    except Exception as e:
        # Make sure we still surface errors when running under MPI,
        # but don't crash the run on a bad file read.
        print(e, file=sys.stderr)
        return None

__license__ = """
 File: test_variables.py
 
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

__doc__ = """\
Test the default variables don't error, and (where possible) that they make sense.
Example numbers taken from Illinois docs wiki: https://github.com/AFD-Illinois/docs/wiki/fmks-and-you
"""

import warnings

import numpy as np

import pyharm
from pyharm.fluid_state import FluidState
from pyharm.grid import Grid

from common import compare


# Parameters for FMKS from a particular simulation,
# MAD a+0.9375 384x192x192 iharm3D eht_v1 run
params = {'coordinates': 'fmks', 'a': 0.9375,
          'r_in': 1.2175642950007606, 'r_out': 1000.0,
          'hslope': 0.3, 'mks_smooth': 0.5, 'poly_xt': 0.82, 'poly_alpha': 14.0,
          'n1': 384, 'n2': 192, 'n3': 192,
          'gam': 13./9, 'gam_e': 4./3, 'gam_p': 5./3,
          'eta': 0.01} # This is only a testing value for viscous Bondi stuff, not physical

# Make a grid
G = Grid(params)

# Values from that simulation, zone 11,12,13 of dump 1200
shape = (1,1,1)
cache = {}
cache['rho'] = 0.42640594 * np.ones(shape)
cache['u']   = 0.18369143 * np.ones(shape)
cache['uvec'] = np.array((0.45536834,  0.01478980,  0.71970361)).reshape(3,*shape)
cache['B']    = np.array((3.66124153,  0.21974833, -5.54809475)).reshape(3,*shape)
# Add some values to test computing other vars
cache['jcon'] = np.array((0, 0, 0, 0)).reshape(4,*shape)
# This ought to behave like any other FluidState, but much less computationally intensive
state = FluidState(cache, params=params, grid=G[11,12,13])
# And set a scale
state.set_units(6.2e9, 1e17)

# This is only for testing divB functions, for now
cache['cons.B'] = state['gdet']*state['B']

def test_fourvs():
    assert compare(state['ucon'],
                   np.array([  2.02058894, -0.26000907, -0.01550771,  0.71970361 ]).reshape(4,1,1,1), rel=1e-7)
    assert compare(state['ucov'],
                   np.array([ -0.56218003,  1.12413841, -0.29943404,  0.58854419 ]).reshape(4,1,1,1), rel=1e-7)
    assert compare(state['bcon'],
                   np.array([  0.78464317,  1.71099976,  0.10273258, -2.46630283 ]).reshape(4,1,1,1), rel=1e-7)
    assert compare(state['bcov'],
                   np.array([  4.16762915,  12.7343388,  2.01595804, -7.05673667 ]).reshape(4,1,1,1), rel=1e-7)

def test_compute_all_vars():
    # Computing all these will generate warnings about numbers
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for var in pyharm.variables.fns_dict:
            assert state[var].shape[-3:] == (1,1,1)

        for vec in ['u^', 'u_', 'b^', 'b_']:
            for i in range(4):
                assert state[vec+str(i)].shape[-3:] == (1,1,1)
            for i in ['t', 'r', 'th', 'phi']:
                assert state[vec+i].shape[-3:] == (1,1,1)
            for i in ['x', 'y', 'z']:
                assert state[vec+i].shape[-3:] == (1,1,1)

        for ten in ['T', 'F']:
            # Contravariant
            for i in ['^0', '^1', '^2', '^3']:
                for j in ['^0', '^1', '^2', '^3']:
                    assert state[ten+i+j].shape[-3:] == (1,1,1)
            # Covariant
            for i in ['_0', '_1', '_2', '_3']:
                for j in ['_0', '_1', '_2', '_3']:
                    assert state[ten+i+j].shape[-3:] == (1,1,1)

        for ten in ['T']:
            # Mixed (only T supported, only first index up)
            for i in ['^0', '^1', '^2', '^3']:
                for j in ['_0', '_1', '_2', '_3']:
                    assert state[ten+i+j].shape[-3:] == (1,1,1)

    # TODO check values against manual or previous computation (GOLD FILES!)
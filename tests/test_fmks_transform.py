__license__ = """
 File: test_fmks_transform.py
 
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
Test our geometry functions make sense.  Example numbers taken from Illinois docs wiki:
https://github.com/AFD-Illinois/docs/wiki/fmks-and-you
"""
import numpy as np

from pyharm.grid import Grid
from pyharm.defs import Loci

from common import compare

# Values in FMKS from a MAD simulation
# zone 11,12,13 of dump 1200 of MAD a+0.9375 384x192x192 iharm3D eht_v1 run
P = np.array([0, 0, 0.4553683,  0.0147898,  0.7197036, 3.6612415,  0.2197483, -5.5480947])

# Metric parameters from that simulation
params = {'coordinates': 'fmks', 'a': 0.9375,
          'r_in': 1.2175642950007606, 'r_out': 1000.0,
          'hslope': 0.3, 'mks_smooth': 0.5, 'poly_xt': 0.82, 'poly_alpha': 14.0,
          'n1': 384, 'n2': 192, 'n3': 192}

# Make a grid
G = Grid(params)

def test_fmks_functions():
    # Derived FMKS metric parameter poly_norm
    assert compare(G.coords.poly_norm, 0.7578173169894967)

    # Zone location in KS
    X = G.coord(11,12,13)
    rhp = np.squeeze(G.coords.ks_coord(X))
    assert compare(rhp, np.array([1.488590864996909, 0.7666458987406977, 0.4417864669110646]))
    # Also test the __getitem__ versions we would call through to from a FluidState
    # Note 'r' and 'th' are length-1 in the last index to save memory & are intended to be broadcast
    rhp = np.array((G['r'][11,12,0], G['th'][11,12,0], G['phi'][11,12,13]))
    assert compare(rhp, np.array([1.488590864996909, 0.7666458987406977, 0.4417864669110646]))

    # Metric values:
    gcov_computed = np.squeeze(G.coords.gcov(X))
    gcov_example = np.array([[ 0.11428415,  1.65871321,  0.        , -0.5027359 ],
                            [ 1.65871321,  4.8045105 , -2.82071735, -1.41998137],
                            [ 0.        , -2.82071735, 66.60209297,  0.        ],
                            [-0.5027359 , -1.41998137,  0.        ,  1.71620473]])
    assert compare(gcov_computed, gcov_example)
    gcov_computed = G['gcov'][:, :, 11,12,0]
    assert compare(gcov_computed, gcov_example)

    gcon_computed = np.squeeze(G.coords.gcon(X))
    gcon_example = np.array([[-2.11428415e+00,  7.48549636e-01,  3.17024113e-02, -4.28728014e-17],
                            [ 7.48549636e-01,  1.98677175e-02,  8.41433249e-04,  2.35714628e-01],
                            [ 3.17024113e-02,  8.41433249e-04,  1.50501794e-02,  9.98293464e-03],
                            [-1.42468631e-17,  2.35714628e-01,  9.98293464e-03,  7.77710464e-01]])
    assert compare(gcon_computed, gcon_example)
    gcon_computed = G['gcon'][:, :, 11,12,0]
    assert compare(gcon_computed, gcon_example)

    # TODO gcon_ks before vs after

# TODO test:
# Making a dump from P and evaluating should give:
# dump['ucon'] = [  2.02058894, -0.26000907, -0.01550771,  0.71970361 ]
# dump['ucov'] = [ -0.56218003,  1.12413841, -0.29943404,  0.58854419 ]
# dump['bcon'] = [  0.78464317,  1.71099976,  0.10273258, -2.46630283 ]
# dump['bcov'] = [  4.16762915,  12.7343388,  2.01595804, -7.05673667 ]
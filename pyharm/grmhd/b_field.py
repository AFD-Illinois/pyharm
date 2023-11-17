__license__ = """
 File: b_field.py
 
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

import numpy as np

from pyharm.defs import Loci, Slices

__doc__ = \
"""Magnetic field tools. Currently just divB.
"""

# TODO flux_ct in numpy, to take a step for jcon

def divB(G, B):
    gdet = G['gdet']

    # If we don't have ghost zones, make our own slices
    s = Slices(ng=1)

    original_shape = B.shape

    divB = np.abs(0.25 * (
            B[0][s.b, s.b, s.b] * gdet[s.b, s.b, :]
            + B[0][s.b, s.l1, s.b] * gdet[s.b, s.l1, :]
            + B[0][s.b, s.b, s.l1] * gdet[s.b, s.b, :]
            + B[0][s.b, s.l1, s.l1] * gdet[s.b, s.l1, :]
            - B[0][s.l1, s.b, s.b] * gdet[s.l1, s.b, :]
            - B[0][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, :]
            - B[0][s.l1, s.b, s.l1] * gdet[s.l1, s.b, :]
            - B[0][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, :]
            ) / G.dx[1] + 0.25 * (
            B[1][s.b, s.b, s.b] * gdet[s.b, s.b, :]
            + B[1][s.l1, s.b, s.b] * gdet[s.l1, s.b, :]
            + B[1][s.b, s.b, s.l1] * gdet[s.b, s.b, :]
            + B[1][s.l1, s.b, s.l1] * gdet[s.l1, s.b, :]
            - B[1][s.b, s.l1, s.b] * gdet[s.b, s.l1, :]
            - B[1][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, :]
            - B[1][s.b, s.l1, s.l1] * gdet[s.b, s.l1, :]
            - B[1][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, :]
            ) / G.dx[2] + 0.25 * (
            B[2][s.b, s.b, s.b] * gdet[s.b, s.b, :]
            + B[2][s.b, s.l1, s.b] * gdet[s.b, s.l1, :]
            + B[2][s.l1, s.b, s.b] * gdet[s.l1, s.b, :]
            + B[2][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, :]
            - B[2][s.b, s.b, s.l1] * gdet[s.b, s.b, :]
            - B[2][s.b, s.l1, s.l1] * gdet[s.b, s.l1, :]
            - B[2][s.l1, s.b, s.l1] * gdet[s.l1, s.b, :]
            - B[2][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, :]
            ) / G.dx[3])

    divB_full = np.zeros(original_shape[1:])
    divB_full[s.b, s.b, s.b] = divB

    return divB_full

def divB_cons(G, B):

    s = Slices(ng=1)

    divB = np.abs(0.25 * (
            B[0][s.b, s.b, s.b]
            + B[0][s.b, s.l1, s.b]
            + B[0][s.b, s.b, s.l1]
            + B[0][s.b, s.l1, s.l1]
            - B[0][s.l1, s.b, s.b]
            - B[0][s.l1, s.l1, s.b]
            - B[0][s.l1, s.b, s.l1]
            - B[0][s.l1, s.l1, s.l1]
            ) / G.dx[1] + 0.25 * (
            B[1][s.b, s.b, s.b]
            + B[1][s.l1, s.b, s.b]
            + B[1][s.b, s.b, s.l1]
            + B[1][s.l1, s.b, s.l1]
            - B[1][s.b, s.l1, s.b]
            - B[1][s.l1, s.l1, s.b]
            - B[1][s.b, s.l1, s.l1]
            - B[1][s.l1, s.l1, s.l1]
            ) / G.dx[2] + 0.25 * (
            B[2][s.b, s.b, s.b]
            + B[2][s.b, s.l1, s.b]
            + B[2][s.l1, s.b, s.b]
            + B[2][s.l1, s.l1, s.b]
            - B[2][s.b, s.b, s.l1]
            - B[2][s.b, s.l1, s.l1]
            - B[2][s.l1, s.b, s.l1]
            - B[2][s.l1, s.l1, s.l1]
            ) / G.dx[3])

    divB_full = np.zeros(B.shape[1:])
    divB_full[s.b, s.b, s.b] = divB

    return divB_full
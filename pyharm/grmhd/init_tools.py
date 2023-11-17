__license__ = """
 File: init_tools.py
 
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

__doc__ = \
"""Tools for translating analytic solutions to KS coordinates.
Eventually, full initialization code to generate a Fishbone-Moncrief torus.
"""

def set_fourvel_t(gcov, ucon):
    AA = gcov[0][0]
    BB = 2. * (gcov[0][1] * ucon[1] + \
               gcov[0][2] * ucon[2] + \
               gcov[0][3] * ucon[3])
    CC = 1. + gcov[1][1] * ucon[1] * ucon[1] + \
         gcov[2][2] * ucon[2] * ucon[2] + \
         gcov[3][3] * ucon[3] * ucon[3] + \
         2. * (gcov[1][2] * ucon[1] * ucon[2] + \
               gcov[1][3] * ucon[1] * ucon[3] + \
               gcov[2][3] * ucon[2] * ucon[3])

    discr = BB * BB - 4. * AA * CC
    ucon[0] = (-BB - np.sqrt(discr)) / (2. * AA)

def fourvel_to_prim(gcon, ucon):
    alpha2 = -1.0 / gcon[0][0]
    # Note gamma/alpha is ucon[0]
    u_prim = np.zeros([3, *ucon.shape[1:]])
    u_prim[0] = ucon[1] + ucon[0] * alpha2 * gcon[0][1]
    u_prim[1] = ucon[2] + ucon[0] * alpha2 * gcon[0][2]
    u_prim[2] = ucon[3] + ucon[0] * alpha2 * gcon[0][3]
    return u_prim
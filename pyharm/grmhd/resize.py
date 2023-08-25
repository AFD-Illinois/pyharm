__license__ = """
 File: resize.py
 
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
from scipy.interpolate import RegularGridInterpolator

from pyharm.defs import Loci
from pyharm.grid import Grid

__doc__ = \
"""Resizing output files. WIP.
"""

def resize_var(params, G, var, n1, n2, n3, method='linear'):
    """Resize a (scalar) variable onto a new grid.
    Note this doesn't yet support ghost zones

    :param G: existing grid
    :param var: variable of correct size

    :returns resized variable
    """
    vnew = np.zeros((n1, n2, n3))
    params_new = params.copy()
    params_new['n1tot'] = params_new['n1'] = n1
    params_new['n2tot'] = params_new['n2'] = n2
    params_new['n3tot'] = params_new['n3'] = n3
    Gnew = Grid(params_new)
    Xnew = Gnew.coord_all()

    # TODO this can be done without coord_all
    X = G.coord_all()
    if len(X.shape) == 4:
        points = (X[1][:,0,0], X[2][0,:,0], X[3][0,0,:])
        interp = RegularGridInterpolator(points, var, method=method, bounds_error=False)
        vnew = interp(Xnew[1:].T).T
    else:
        points = (X[1][:,0], X[2][0,:])
        interp = RegularGridInterpolator(points, np.squeeze(var), method=method, bounds_error=False)
        vnew = interp(Xnew[1:3].T).T
    del X, Xnew, points

    return vnew, Gnew

def resize(params, G, P, n1, n2, n3, method='linear'):
    """Resize the primitives P onto a new grid.
    Note this doesn't yet support ghost zones
    """
    nvar = P.shape[0]
    X = G.coord_all()

    Pnew = np.zeros((nvar, n1, n2, n3))
    params_new = params.copy()
    params_new['n1tot'] = params_new['n1'] = n1
    params_new['n2tot'] = params_new['n2'] = n2
    params_new['n3tot'] = params_new['n3'] = n3
    Gnew = Grid(params_new)
    Xnew = Gnew.coord_all()

    for var in range(nvar):
        interp = RegularGridInterpolator((X[1][:,0,0], X[2][0,:,0], X[3][0,0,:]), P[var], method=method, bounds_error=False)
        points = interp(Xnew[1:].T)
        Pnew[var] = points.T

    return params_new, Gnew, Pnew
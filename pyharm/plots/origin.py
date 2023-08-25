__license__ = """
 File: origin.py
 
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

from .plot_utils import *

__doc__ = \
"""Plots related to zone-based data recorded by e.g. ipole, not a full fluid dump.
The primary function plots data recorded by ipole in "histo" mode, which records any
intensity changes in a bin corresponding to the nearest zone.  This illustrates where
observed emission tends to originate.
The other functions plot properties of the spacetime or illustrate geometry such as the
observer angle.
"""

def plot_emission_origin(ax, grid, Inu, window=None, sz=6, log=False, vmin=None, vmax=None):
    """Plot the origin of emission, as recorded by ipole in a 3D histogram of emission count/zone Inu.
    Normalizes by zone area on the plot in order to show true emission density.
    """
    var = np.mean(Inu, axis=-1) / grid.get_xz_areas(half_cut=True)
    X, Y = grid.get_xz_locations(mesh=True, half_cut=True)

    pargs = {'cmap':'afmhot', 'shading':'flat', 'linewidth':0,
             'vmin':vmin, 'vmax':vmax, 'rasterized':True}

    ax.set_facecolor('black')
    if log:
        pcol = pcolormesh_log(ax, X, Y, var, **pargs)
    else:
        pcol = ax.pcolormesh(X, Y, var, **pargs)

    if window is None:
        window = [0, 2*sz, -sz, sz]
    ax.set_xlim(window[:2])
    ax.set_ylim(window[2:])

    ax.set_aspect('equal')
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.set_xlabel(r"$x\;[GM/c^2]$",fontsize=14)
    ax.set_ylabel(r"$z\;[GM/c^2]$",fontsize=14)
    ax.tick_params(axis='x', which='both',direction='inout', bottom=True, labelbottom=True, labelsize=14)
    ax.tick_params(axis='y', which='both',direction='inout', left=True, labelleft=True, labelsize=14)

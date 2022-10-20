__license__ = """
 File: origin.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020-2022, AFD Group at UIUC
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
from matplotlib.patches import Circle

__doc__ = \
"""Plots related to zone-based data recorded by e.g. ipole, not a full fluid dump.
The primary function plots data recorded by ipole in "histo" mode, which records any
intensity changes in a bin corresponding to the nearest zone.  This illustrates where
observed emission tends to originate.
The other functions plot properties of the spacetime or illustrate geometry such as the
observer angle.
"""

def plot_emission_origin(ax, grid, Inu, window=None, sz=6):
    """Plot the origin of emission, as recorded by ipole in a 3D histogram of emission count/zone Inu.
    Normalizes by zone area on the plot in order to show true emission density.
    """
    var = np.mean(Inu, axis=-1) / grid.get_xz_areas(half_cut=True)
    X, Y = grid.get_xz_locations(mesh=True, half_cut=True)

    ax.set_facecolor('black')
    pcol = ax.pcolormesh(X, Y, var, cmap='afmhot', shading='flat', linewidth=0, rasterized=True)
    pcol.set_edgecolor('face')

    if window is None:
        window = [0, 2*sz, -sz, sz]
    ax.set_xlim(window[:2])
    ax.set_ylim(window[2:])

    ax.set_aspect('equal')
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.set_xlabel("$x [GM/c^2]$",fontsize=14)
    ax.set_ylabel("$z [GM/c^2]$",fontsize=14)
    ax.tick_params(axis='x', which='both',direction='inout', bottom=True, labelbottom=True, labelsize=14)
    ax.tick_params(axis='y', which='both',direction='inout', left=True, labelleft=True, labelsize=14)

def overlay_eh_border(ax, grid, at_i=5, color='#FFFFFF', linewidth=1.5):
    """Plot a circle corresponding to the `i`th zone of the simulation.
    Usually used for the EH at i=5th zone.
    """
    X, Y = grid.get_xz_locations(mesh=True, half_cut=True)
    ax.plot(X[at_i,:], Y[at_i,:], color, linewidth=linewidth) #white

def overlay_circle(ax, at_r=10, color='#FFFFFF'):
    """Plot a circle at radius 'at_r' in M from the origin."""
    ax.add_patch(Circle((0,0), radius=at_r, edgecolor=color, facecolor='none'))

def overlay_photon_orbits(ax, a):
    """Overlay a region representing allowable photon orbits given the black hole spin"""
    def uplus(a,r):
        # return allowed photon orbit locations, credit Leo C. Stein
        #   https://duetosymmetry.com/tool/kerr-circular-photon-orbits/
        Q = - r**3*(r**3-6.*r*r+9.*r-4.*a*a)/a/a/(r-1.)/(r-1.)
        Phi = - (r**3 - 3.*r*r+a*a*r+a*a) / a / (r - 1.)
        return np.sqrt(1./2/a/a*( (a*a-Q-Phi*Phi) + np.sqrt( (a*a-Q-Phi*Phi)**2 + 4*a*a*Q) ))
    rs = np.linspace(1.,4.,100000)
    up = uplus(a,rs)
    xs = rs * np.sin(np.arccos(up))
    yps = rs * up
    yms = - rs * up
    ax.plot(xs,yps,'w--',linewidth=2)
    ax.plot(xs,yms,'w--',linewidth=2)

def overlay_observer_arrow(ax, angle=163, r_start=4, r_end=5.5):
    """Place an arrow at 'angle' marking the direction to the observer"""
    th = angle/180.*np.pi
    x1 = r_start * np.sin(angle)
    y1 = r_start * np.cos(angle)
    x2 = r_end * np.sin(angle)
    y2 = r_end * np.cos(angle)
    ax.arrow(x1,y1, (x2-x1),(y2-y1), head_width=0.4, head_length=0.4, fc='#00ff00', ec='#00ff00', width=0.1)

def mark_isco(ax, grid, color='#00FF00'):
    """Place an x at the ISCO radius in the midplane of a plot."""
    ax.plot(grid['r_isco'], 0, 'x', color=color, ms=8, mew=3)
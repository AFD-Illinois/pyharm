__license__ = """
 File: overlays.py
 
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

from scipy.integrate import trapz

from matplotlib.patches import Circle

from ..ana.reductions import flatten_xy, flatten_xz, wrap
from .plot_utils import *

#### GEOMETRY ####

def overlay_circle(ax, at_r=10, color='#FFFFFF'):
    """Plot a circle at radius 'at_r' in M from the origin."""
    ax.add_patch(Circle((0,0), radius=at_r, edgecolor=color, facecolor='none'))

def overlay_observer_arrow(ax, angle=163, r_start=4, r_end=5.5):
    """Place an arrow at 'angle' marking the direction to the observer"""
    th = angle/180.*np.pi
    x1 = r_start * np.sin(th)
    y1 = r_start * np.cos(th)
    x2 = r_end * np.sin(th)
    y2 = r_end * np.cos(th)
    ax.arrow(x1,y1, (x2-x1),(y2-y1), head_width=0.4, head_length=0.4, fc='#00ff00', ec='#00ff00', width=0.1)

#### SPACETIME ####

def overlay_eh_border(ax, grid, at_i=5, color='#FFFFFF', linewidth=1.5):
    """Plot a circle corresponding to the `i`th zone of the simulation.
    Usually used for the EH at i=5th zone.
    """
    X, Y = grid.get_xz_locations(mesh=True, half_cut=True)
    ax.plot(X[at_i,:], Y[at_i,:], color, linewidth=linewidth) #white

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

def mark_isco(ax, grid, color='#00FF00'):
    """Place an x at the ISCO radius in the midplane of a plot."""
    ax.plot(grid['r_isco'], 0, 'x', color=color, ms=8, mew=3)

#### GRID ####

def overlay_grid(ax, grid, color='k', linewidth=0.2):
    c = grid.coords
    m = grid.coord_all(mesh=True)
    for i in range(grid['n1']):
        ax.plot(c.cart_x(m[:,:,i,0]), c.cart_z(m[:,:,i,0]), color=color, linewidth=linewidth)
    for i in range(grid['n2']):
        ax.plot(c.cart_x(m[:,i,:,0]), c.cart_z(m[:,i,:,0]), color=color, linewidth=linewidth)

#### VARIABLES ####

def overlay_contours(ax, dump, var, levels, color='k', native=False, half_cut=False, at=0, average=False, use_contourf=False, **kwargs):
    """Overlay countour lines on an XZ plot, of 'var' at each of the list 'levels' in color 'color'.
    Takes a bunch of the same other options as the plot it's overlaid upon.
    """
    # TODO optional line cutoff by setting NaN according to a second condition
    # TODO these few functions could just optionally use np.mean
    x, z = dump.grid.get_xz_locations(native=native, half_cut=(half_cut or native))
    if average:
        var = np.squeeze(flatten_xz(dump, var, at, True) / dump['n3']) # Arg to flatten_xz is "sum", so we divide
    else:
        var = np.squeeze(flatten_xz(dump, var, at, False))
    if use_contourf:
        return ax.contourf(x, z, var, levels=levels, colors=color, **kwargs)
    else:
        return ax.contour(x, z, var, levels=levels, colors=color, **kwargs)

def overlay_field(ax, dump, **kwargs):
        overlay_flowlines(ax, dump, 'B1', 'B2', **kwargs)

def overlay_flowlines(ax, dump, varx1, varx2, nlines=20, color='k', native=False, half_cut=False, reverse=False, **kwargs):
    """Overlay the "flow lines" of a pair of variables in X1 and X2 directions.  Sums assuming no divergence to obtain a
    potential, then plots contours of the potential so as to total 'nlines' total contours.
    """

    if native:
        half_cut = True

    x, z = dump.grid.get_xz_locations(native=native, half_cut=half_cut)
    varx1 = flatten_xz(dump, varx1, sum=True, half_cut=True) / dump['n3'] * np.squeeze(dump['gdet'])
    varx2 = flatten_xz(dump, varx2, sum=True, half_cut=True) / dump['n3'] * np.squeeze(dump['gdet'])

    if native:
        varx1 = varx1.T
        varx2 = -varx2.T

    N1, N2 = varx1.shape[:2]

    AJ_phi = np.zeros([N1, 2*N2])
    for j in range(N2):
        for i in range(N1):
            if not reverse:
                AJ_phi[i, N2 - 1 - j] = AJ_phi[i, N2 + j] = \
                    (trapz(varx2[:i, j], dx=dump['dx1']) -
                     trapz(varx1[i, :j], dx=dump['dx2']))
            else:
                AJ_phi[i, N2 - 1 - j] = AJ_phi[i, N2 + j] = \
                    (trapz(varx2[:i, j], dx=dump['dx1']) +
                     trapz(varx1[i, j:], dx=dump['dx2']))
    AJ_phi -= AJ_phi.min()
    levels = np.linspace(0, AJ_phi.max(), nlines * 2)

    if half_cut:
        AJ_phi = AJ_phi[:,:N2]

    ax.contour(x, z, AJ_phi, levels=levels, colors=color)


def overlay_quiver(ax, dump, varx1, varx2, cadence=64, norm=1):
    """Overlay a quiver plot of 2 vector components onto a plot in *native coordinates only*."""
    varx1 = flatten_xz(dump, varx1, sum=True) / dump['n3'] * dump['gdet']
    varx2 = flatten_xz(dump, varx2, sum=True) / dump['n3'] * dump['gdet']
    max_J = np.max(np.sqrt(varx1 ** 2 + varx2 ** 2))

    x, z = dump.grid.get_xz_locations(native=True)

    s1 = dump['n1'] // cadence
    s2 = dump['n2'] // cadence

    ax.quiver(x[::s1, ::s2], z[::s1, ::s2], varx1[::s1, ::s2], varx2[::s1, ::s2],
              units='xy', angles='xy', scale_units='xy', scale=(cadence * max_J / norm))


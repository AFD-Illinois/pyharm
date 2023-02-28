__license__ = """
 File: plot_dumps.py
 
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

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from ..ana.reductions import flatten_xy, flatten_xz, wrap
from ..util import i_of
from .plot_utils import *
from .overlays import *
from .pretty import pretty

__doc__ = \
"""2D plots of variables over different slices of a simulation domain.
This file handles plotting of any particular variable on a particular ``matplotlib.Axis`` object.
For full figures with default annotations, variable choices, etc, see ``figures.py``
"""

def _decorate_plot(ax, dump, var, bh=True, xticks=None, yticks=None, frame=True,
                  cbar=True, cbar_ticks=None, cbar_label=None,
                  label=None, **kwargs):
    """Add any extras to plots which are not dependent on data or slicing.
    Accepts arbitrary extra arguments for compatibility -- they are passed nowhere.
    
    :param bh: Add the BH silhouette
    
    :param xticks: If not None, set *all* xticks to specified list
    :param yticks: If not None, set *all* yticks to specified list
    :param frame: whether to show the plot axes/border at all
    
    :param cbar: Add a colorbar
    :param cbar_ticks: If not None, set colorbar ticks to specified list
    :param cbar_label: If not None, set colorbar label
    
    :param label: If not None, set plot title
    """

    if bh and ("minkowski" not in dump['coordinates']) and ("cartesian" not in dump['coordinates']):
        # BH silhouette
        circle1 = plt.Circle((0, 0), dump['r_eh'], color='k')
        ax.add_artist(circle1)

    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(ax.collections[0], cax=cax)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
        if cbar_label is not None:
            cbar.set_label(cbar_label)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if not frame:
        # Remove the whole frame for good measure
        ax.axis('off')
        # What does this do? It looks scary
        # fig.patch.set_visible(False)

    if label is not None:
        ax.set_title(label)
    elif isinstance(var, str):
        ax.set_title(pretty(var))

def plot_xz(ax, dump, var, vmin=None, vmax=None, window=(-40, 40, -40, 40),
            xlabel=True, ylabel=True, native=False, log=False,
            half_cut=False, cmap='jet', shading='gouraud',
            at=None, average=False, sum=False, cbar=True, log_r=False, **kwargs):
    """Plot a poloidal or X1/X2 slice of a dump file.
    Note this function also accepts all keyword arguments to _decorate_plot()

    :param ax: Axes object to paint on
    :param dump: fluid state object
    :param vmin, vmax: colorbar minimum and maximum
    :param window: view window in X,Z coordinates, measured in r_g/c^2, 0 in BH center.
    :param xlabel, ylabel: whether to mark X/Y labels with reasonable titles
    :param native: Plot in native coordinates X1/X2 as plot X/Y axes respectively
    :param log: plot a signed quantity in logspace with symlog() above
    """

    vname = None
    if isinstance(var, str):
        if 'symlog_' in var:
            log = True
            var = var.replace("symlog_","")
        elif 'log_' in var:
            log = True
            var = var.replace("log_","")
        vname = var

    x, z = dump.grid.get_xz_locations(mesh=(shading == 'flat'), native=native, half_cut=(half_cut or native), log_r=log_r)
    var = flatten_xz(dump, var, at, sum or average, half_cut or native)
    if average:
        var /= dump['n3']
    if shading != 'flat':
        x = wrap(x)
        z = wrap(z)
        var = wrap(var)

    # Use symlog only when we need it
    if log and np.any(var < 0.0):
        if cmap == 'jet':
            cmap = 'RdBu_r'
        mesh = pcolormesh_symlog(ax, x, z, var, cmap=cmap, vmin=vmin, vmax=vmax,
                                 shading=shading, cbar=cbar) # Use this cbar, it's customized
        cbar = False # We don't need another later on
    elif log:
        # Support legacy calling convention
        if vmin is not None and vmin < 0:
            vmin = 10**vmin
            vmax = 10**vmax
        mesh = pcolormesh_log(ax, x, z, var, cmap=cmap, vmin=vmin, vmax=vmax,
                              shading=shading, cbar=cbar) # Use this cbar, it's customized
        cbar = False # We don't need another later on
    else:
        mesh = ax.pcolormesh(x, z, var, cmap=cmap, vmin=vmin, vmax=vmax,
                             shading=shading)

    if native:
        if xlabel: ax.set_xlabel("X1 (native coordinates)")
        if ylabel: ax.set_ylabel("X2 (native coordinates)")
        if window:
            ax.set_xlim(window[:2])
            ax.set_ylim(window[2:])
        else:
            # Just set to th
            ax.set_xlim([np.min(x), np.max(x)])
            ax.set_ylim([np.min(z), np.max(z)])
    elif log_r:
        if xlabel: ax.set_xlabel(r"$x$ ($r \rightarrow \log_{10}(r)$)")
        if ylabel: ax.set_ylabel(r"$z$ ($r \rightarrow \log_{10}(r)$)")
        if window is not None:
            ax.set_xlim(window[:2])
            ax.set_ylim(window[2:])
        else:
            ax.set_xlim([np.min(x), np.max(x)])
            ax.set_ylim([np.min(z), np.max(z)])
    else:
        if xlabel: ax.set_xlabel(r"$x$ ($r_g$)")
        if ylabel: ax.set_ylabel(r"$z$ ($r_g$)")
        if window:
            ax.set_xlim(window[:2])
            ax.set_ylim(window[2:])
        # TODO alt option of size -r_out to r_out?

    # TODO do we ever not want this?
    ax.set_aspect('equal')

    # Set up arguments for decorating plot
    if not 'bh' in kwargs:
        kwargs['bh'] = not native

    # Restore "var" to string for labelling plots
    if vname is not None:
        var = vname
        if log:
            var = "log_"+var
    _decorate_plot(ax, dump, var, cbar=cbar, **kwargs)

    # In case user wants to tweak this
    return mesh

def plot_xy(ax, dump, var, vmin=None, vmax=None, window=None,
            xlabel=True, ylabel=True, native=False, log=False,
            cmap='jet', shading='gouraud',
            at=None, average=False, sum=False, cbar=True, log_r=True, **kwargs):
    """Plot a toroidal or X1/X3 slice of a dump file.
    Note this function also accepts all keyword arguments to _decorate_plot()

    :param ax: Axes object to paint on
    :param dump: fluid state object
    :param vmin, vmax: colorbar minimum and maximum, 'None' auto-detects
    :param window: view window in X,Z coordinates, measured in r_g/c^2, 0 in BH center.
    :param xlabel, ylabel: whether to mark X/Y labels with reasonable titles
    :param native: Plot in native coordinates X1/X2 as plot X/Y axes respectively
    :param log: plot a signed quantity in logspace with symlog() above
    """


    # This plot makes no sense for 2D dumps yet gets called a bunch in movies
    if dump['n3'] == 1:
        return None

    vname = None
    if isinstance(var, str):
        if 'symlog_' in var:
            log = True
            var = var.replace("symlog_","")
        elif 'log_' in var:
            log = True
            var = var.replace("log_","")
        vname = var

    x, y = dump.grid.get_xy_locations(mesh=(shading == 'flat'), native=native, log_r=log_r)
    var = flatten_xy(dump, var, at, sum or average)
    if average:
        var /= dump['n2']
    if shading != 'flat':
        x = wrap(x)
        y = wrap(y)
        var = wrap(var)

    # Use symlog only when we need it
    # TODO test only inside our window?
    if log and np.any(var < 0.0):
        if cmap == 'jet':
            cmap = 'RdBu_r'
        mesh = pcolormesh_symlog(ax, x, y, var, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading=shading, cbar=cbar)
        cbar = False
    elif log:
        # Support legacy calling convention
        if vmin is not None and vmin < 0:
            vmin = 10**vmin
            vmax = 10**vmax
        mesh = pcolormesh_log(ax, x, y, var, cmap=cmap, vmin=vmin, vmax=vmax,
                              shading=shading, cbar=cbar) # Use this cbar, it's customized
        cbar = False # We don't need another later on
    else:
        mesh = ax.pcolormesh(x, y, var, cmap=cmap, vmin=vmin, vmax=vmax,
                         shading=shading)

    if native:
        if xlabel: ax.set_xlabel("X1 (native coordinates)")
        if ylabel: ax.set_ylabel("X3 (native coordinates)")
        if window is not None:
            ax.set_xlim(window[:2])
            ax.set_ylim(window[2:])
        else:
            ax.set_xlim([x[0,0], x[-1,-1]])
            ax.set_ylim([y[0,0], y[-1,-1]])
    elif log_r:
        if xlabel: ax.set_xlabel(r"$x$ ($r \rightarrow \log_{10}(r)$)")
        if ylabel: ax.set_ylabel(r"$y$ ($r \rightarrow \log_{10}(r)$)")
        if window is not None:
            ax.set_xlim(window[:2])
            ax.set_ylim(window[2:])
        else:
            ax.set_xlim([np.min(x), np.max(x)])
            ax.set_ylim([np.min(y), np.max(y)])
    else:
        if xlabel: ax.set_xlabel(r"$x$ ($r_g$)")  # or \frac{G M}{c^2}
        if ylabel: ax.set_ylabel(r"$y$ ($r_g$)")
        if window is not None:
            ax.set_xlim(window[:2])
            ax.set_ylim(window[2:])
        else:
            # TODO guess this?
            pass

    # TODO do we ever not want this?
    ax.set_aspect('equal')

    # Set up arguments for decorating plot
    if not 'bh' in kwargs:
        kwargs['bh'] = not native

    # Restore "var" to string for labelling plots
    if vname is not None:
        var = vname
        if log:
            var = "log_"+var
    _decorate_plot(ax, dump, var, cbar=cbar, **kwargs)

    # In case user wants to tweak this
    return mesh


def plot_thphi(ax, dump, var, at_r=None, at_i=None, cmap='jet', vmin=None, vmax=None, window=None,
               xlabel=True, ylabel=True, native=False, log=False,
               projection='mercator', shading='gouraud', **kwargs):
    """Plot a theta-phi slice at index at_i
    Note this function also accepts all keyword arguments to _decorate_plot()

    :param ax: Axes object to paint on
    :param dump: fluid state object
    :param vmin, vmax: colorbar minimum and maximum, 'None' auto-detects
    :param window: view window in X,Z coordinates, measured in r_g/c^2, 0 in BH center.
    :param xlabel, ylabel: whether to mark X/Y labels with reasonable titles
    :param native: Plot in native coordinates X1/X2 as plot X/Y axes respectively
    :param log: plot a signed quantity in logspace with symlog() above
    :param projection:
        | "mercator": default, project theta on Y-axis and phi on X-axis. Differs from 'native' due to midplane compression.
        | "polar": view down from +z.  Or with 'bottom', view up from -Z.
        | "flattened_polar": reinterpret as polar coordinates, theta -> r, phi -> phi
    :param shading: 
    """
    if at_i is not None:
        at = at_i
    elif at_r is not None:
        at = i_of(dump['r1d'], at_r)
    else:
        raise ValueError("Must specify radial location for solid-angle plot!")

    vname = None
    if isinstance(var, str):
        if 'symlog_' in var:
            log = True
            var = var.replace("symlog_","")
        vname = var
        var = np.squeeze(dump[at, :, :][var])
    else:
        var = var[at, :, :]

    x, y = dump.grid.get_thphi_locations(mesh=(shading == 'flat'), native=native, at=at, projection=projection)
    if (shading != 'flat') and (projection != 'mercator'):
        x = wrap(x)
        y = wrap(y)
        var = wrap(var)

    mesh = ax.pcolormesh(x, y, var, cmap=cmap, vmin=vmin, vmax=vmax,
                         shading=shading)

    if native:
        if xlabel: ax.set_xlabel("X3 (native coordinates)")
        if ylabel: ax.set_ylabel("X2 (native coordinates)")
        if window is not None:
            ax.set_xlim(window[:2])
            ax.set_ylim(window[2:])
        else:
            ax.set_xlim([x[0,0], x[-1,-1]])
            ax.set_ylim([y[0,0], y[-1,-1]])
    else:
        if xlabel: ax.set_xlabel(r"$\phi$")
        if ylabel: ax.set_ylabel(r"$\theta$")
        if window is not None:
            ax.set_xlim(window[:2])
            ax.set_ylim(window[2:])
        ax.set_aspect('equal')

    # Restore "var" to string for labelling plots
    if vname is not None:
        var = vname
        if log:
            var = "log_"+var
    _decorate_plot(ax, dump, var, bh=False, **kwargs)

    return mesh

def plot_slices(ax1, ax2, dump, var, field_overlay=False, nlines=10, **kwargs):
    """Make adjacent plots with plot_xy and plot_xz, using the given pair of axes.
    Assumes axes are arranged left-to-right ax1, ax2
    """
    kwargs_left = {**kwargs, **{'cbar': False}}
    plot_xz(ax1, dump, var, **kwargs_left)
    # If we're not plotting in native coordinates, plot contours.
    # They are very unintuitive in native coords
    if field_overlay and not ('native' in kwargs.keys() and kwargs['native']):
        overlay_field(ax1, dump, nlines=nlines)

    # If we specified 'at', it was *certainly* for the xz slice, not this one.
    # TODO separate option when calling this/plot_xy that will disambiguate?
    kwargs_right = {**kwargs, **{'at': None}}
    plot_xy(ax2, dump, var, **kwargs_right)

def plot_diff_xy(ax, dump1, dump2, var, rel=False, **kwargs):
    if rel:
        plot_xy(ax, dump1, np.abs((dump1[var] - dump2[var])/dump1[var]),
            label=pretty(var), **kwargs)
    else:
        plot_xy(ax, dump1, np.abs(dump1[var] - dump2[var]),
            log=True, label=pretty(var), **kwargs)

def plot_diff_xz(ax, dump1, dump2, var, rel=False, **kwargs):
    if rel:
        plot_xz(ax, dump1, np.abs((dump1[var] - dump2[var])/dump1[var]),
            label=pretty(var), **kwargs)
    else:
        plot_xz(ax, dump1, np.abs(dump1[var] - dump2[var]),
            log=True, label=pretty(var), **kwargs)
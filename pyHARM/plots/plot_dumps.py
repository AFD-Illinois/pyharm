import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from scipy.integrate import trapz

from ..reductions import flatten_xy, flatten_xz, wrap
from .plot_utils import *
from .pretty import pretty

"""Plots of variables over (parts of) a simulation domain represented by a fluid dump file/object
Can overlay "magnetic field lines" computed as contours of phi-symmetrized field's vector potential
"""

def _decorate_plot(ax, dump, var, bh=True, xticks=None, yticks=None,
                  cbar=True, cbar_ticks=None, cbar_label=None,
                  label=None, **kwargs):
    """Add any extras to plots which are not dependent on data or slicing.
    Accepts arbitrary extra arguments for compatibility -- they are passed nowhere.
    
    :param bh: Add the BH silhouette
    
    :param xticks: If not None, set *all* xticks to specified list
    :param yticks: If not None, set *all* yticks to specified list
    
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
        plt.gca().set_xticks(xticks)
        plt.xticks(xticks)
        ax.set_xticks(xticks)
    if yticks is not None:
        plt.gca().set_yticks(yticks)
        plt.yticks(yticks)
        ax.set_yticks(yticks)
    if xticks == [] and yticks == []:
        # Remove the whole frame for good measure
        # fig.patch.set_visible(False)
        ax.axis('off')

    if label is not None:
        ax.set_title(label)
    elif isinstance(var, str):
        ax.set_title(pretty(var))

def plot_xz(ax, dump, var, vmin=None, vmax=None, window=(-40, 40, -40, 40),
            xlabel=True, ylabel=True, native=False, log=False,
            half_cut=False, cmap='jet', shading='gouraud',
            at=None, average=False, sum=False, **kwargs):
    """Plot a poloidal or X1/X2 slice of a dump file.
    NOTE: also accepts all keyword arguments to _decorate_plot()
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
        vname = var

    x, z = dump.grid.get_xz_locations(mesh=(shading == 'flat'), native=native, half_cut=half_cut)
    var = flatten_xz(dump, var, at, sum or average, half_cut)
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
                                 shading=shading, cbar=False) # We add a colorbar later
    else:
        if log:
            var = np.log10(var)
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
    else:
        if xlabel: ax.set_xlabel(r"x ($r_g$)")
        if ylabel: ax.set_ylabel(r"z ($r_g$)")
        if window:
            ax.set_xlim(window[:2])
            ax.set_ylim(window[2:])
        # TODO alt option of size -r_out to r_out?

        if not half_cut:
            ax.set_aspect('equal')

    # Set up arguments for decorating plot
    if not 'bh' in kwargs:
        kwargs['bh'] = not native

    # Restore "var" to string for labelling plots
    if vname is not None:
        var = vname
        if log:
            var = "log_"+var
    _decorate_plot(ax, dump, var, **kwargs)

    # In case user wants to tweak this
    return mesh

def plot_xy(ax, dump, var, vmin=None, vmax=None, window=None,
            xlabel=True, ylabel=True, native=False, log=False,
            cmap='jet', shading='gouraud',
            at=None, average=False, sum=False, **kwargs):
    """Plot a toroidal or X1/X3 slice of a dump file.
    NOTE: also accepts all keyword arguments to _decorate_plot()
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
        vname = var

    x, y = dump.grid.get_xy_locations(mesh=(shading == 'flat'), native=native)
    var = flatten_xy(dump, var, at, sum or average)
    if average:
        var /= dump['n2']
    if shading != 'flat':
        x = wrap(x)
        y = wrap(y)
        var = wrap(var)

    # Use symlog only when we need it
    if log and np.any(var < 0.0):
        if cmap == 'jet':
            cmap = 'RdBu_r'
        mesh = pcolormesh_symlog(ax, x, y, var, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading=shading, cbar=False) # We add a colorbar later
    else:
        if log:
            var = np.log10(var)
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
    else:
        if xlabel: ax.set_xlabel(r"x ($r_g$)")  # or \frac{G M}{c^2}
        if ylabel: ax.set_ylabel(r"y ($r_g$)")
        if window is not None:
            ax.set_xlim(window[:2])
            ax.set_ylim(window[2:])
        else:
            ax.set_xlim(window[:2])
            ax.set_ylim(window[2:])
        ax.set_aspect('equal')

    # Set up arguments for decorating plot
    if not 'bh' in kwargs:
        kwargs['bh'] = not native

    # Restore "var" to string for labelling plots
    if vname is not None:
        var = vname
        if log:
            var = "log_"+var
    _decorate_plot(ax, dump, var, **kwargs)

    # In case user wants to tweak this
    return mesh


# TODO this is currently just for profiles already in 2D
def plot_thphi(ax, dump, var, r_i, cmap='jet', vmin=None, vmax=None, window=None,
               xlabel=True, ylabel=True, native=False, log=False,
               project=False, shading='gouraud', **kwargs):
    """Plot a theta-phi slice at index r_i
    DO NOT USE YET, DEAD
    """
    if native:
        # X3-X2 makes way more sense than X2-X3 since the disk is horizontal
        x = (dump['X3'][r_i] - dump['startx3']) / (dump['n3'] * dump['dx3'])
        y = (dump['X2'][r_i] - dump['startx2']) / (dump['n2'] * dump['dx2'])
    else:
        radius = dump['r'][r_i, 0, 0]
        max_th = dump['n2'] // 2
        if project:
            x = _flatten_yz(dump['th'] * np.cos(dump['phi']), r_i)[:max_th, :]
            y = _flatten_yz(dump['th'] * np.sin(dump['phi']), r_i)[:max_th, :]
        else:
            x = _flatten_yz(dump['x'], r_i)[:max_th, :]
            y = _flatten_yz(dump['y'], r_i)[:max_th, :]
        var = _flatten_yz(var[:max_th, :])

    if window is None:
        if native:
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        elif project:
            window = [-1.6, 1.6, -1.6, 1.6]
        else:
            window = [-radius, radius, -radius, radius]
    else:
        ax.set_xlim(window[:2])
        ax.set_ylim(window[2:])

    mesh = ax.pcolormesh(x, y, var, cmap=cmap, vmin=vmin, vmax=vmax,
                         shading=shading)

    if native:
        if xlabel: ax.set_xlabel("X3 (arbitrary)")
        if ylabel: ax.set_ylabel("X2 (arbitrary)")
    else:
        if xlabel: ax.set_xlabel(r"$x \frac{c^2}{G M}$")
        if ylabel: ax.set_ylabel(r"$y \frac{c^2}{G M}$")

    ax.set_aspect('equal')
    _decorate_plot(ax, dump, var, bh=False, **kwargs)

    return mesh

def plot_slices(ax1, ax2, dump, var, field_overlay=False, nlines=10, **kwargs):
    """Make adjacent plots with plot_xy and plot_xz, using the given pair of axes
    """
    plot_xz(ax1, dump, var, **kwargs)
    # If we're not plotting in native coordinates, plot contours.
    # They are very unintuitive in native coords
    if field_overlay and not ('native' in kwargs.keys() and kwargs['native']):
        overlay_field(ax1, dump, nlines=nlines)

    # If we specified 'at', it was *certainly* for the xz slice, not this one.
    # TODO separate option when calling this/plot_xy that will disambiguate?
    if 'at' in kwargs:
        kwargs['at'] = None
    plot_xy(ax2, dump, var, **kwargs)

def overlay_contours(ax, dump, var, levels, color='k', native=False, half_cut=False, at=0, average=False, use_contourf=False, **kwargs):
    """Overlay countour lines on an XZ plot, of 'var' at each of the list 'levels' in color 'color'.
    Takes a bunch of the same other options as the plot it's overlaid upon.
    """
    # TODO optional line cutoff by setting NaN according to a second condition
    # TODO these few functions could just optionally use np.mean
    x, z = dump.grid.get_xz_locations(native=native, half_cut=half_cut)
    var = flatten_xz(dump, var, at, average) / dump['n3'] # Arg to flatten_xz is "sum", so we divide
    if use_contourf:
        return ax.contourf(x, z, var, levels=levels, colors=color, **kwargs)
    else:
        return ax.contour(x, z, var, levels=levels, colors=color, **kwargs)

def overlay_field(ax, dump, **kwargs):
        overlay_flowlines(ax, dump, 'B1', 'B2', **kwargs)

def overlay_flowlines(ax, dump, varx1, varx2, nlines=20, color='k', native=False, half_cut=False, reverse=False):
    """Overlay the "flow lines" of a pair of variables in X1 and X2 directions.  Sums assuming no divergence to obtain a
    potential, then plots contours of the potential so as to total 'nlines' total contours.
    """
    N1 = dump['n1']
    N2 = dump['n2']

    x, z = dump.grid.get_xz_locations(native=native, half_cut=half_cut)
    varx1 = flatten_xz(dump, varx1, sum=True, half_cut=True) / dump['n3'] * np.squeeze(dump['gdet'])
    varx2 = flatten_xz(dump, varx2, sum=True, half_cut=True) / dump['n3'] * np.squeeze(dump['gdet'])

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


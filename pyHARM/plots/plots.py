import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from scipy.integrate import trapz

from pyHARM.plots.pretty import pretty

"""Plo
"""

def pcolormesh_symlog(ax, X, Y, Z, vmax=None, vmin=None, linthresh=None, decades=4, linscale=0.01, cmap='RdBu_r', cbar=True, **kwargs):
    """Wrapper for matplotlib's pcolormesh that uses it sensibly, instead of the defaults.

    If linthresh is not specified, it defaults to vmax*10**(-decades), i.e. showing that number of decades each
    of positive and negative values.

    If not specified, vmax is set automatically
    In order to keep colors sensible, vmin is overridden unless set alone.
    """
    if vmax is None:
        if vmin is not None:
            vmax = -vmin
        else:
            vmax = np.abs(np.nanmax(Z))*2
            vmin = -vmax
            #print("Using automatic range {} to {}".format(vmin, vmax))
    else:
        vmin = -vmax

    int_min_pow, int_max_pow = int(np.ceil(np.log10(-vmin))), int(np.ceil(np.log10(vmax)))

    if linthresh is None:
        linthresh = vmax * 10**(-decades)

    logthresh = -int(np.ceil(np.log10(linthresh)))
    tick_locations = ([vmin]
                      + [-(10.0 ** x) for x in range(int_min_pow - 1, -logthresh - 1, -1)]
                      + [0.0]
                      + [(10.0 ** x) for x in range(-logthresh, int_max_pow)]
                      + [vmax])
    pcm = ax.pcolormesh(X, Y, Z, norm=colors.SymLogNorm(linthresh=linthresh, linscale=linscale, base=10, vmin=-vmax, vmax=vmax),
                         cmap=cmap, **kwargs)
    if cbar:
        # TODO add some more anonymous ticks
        plt.colorbar(pcm, ax=ax, ticks=tick_locations, format=ticker.LogFormatterMathtext())

    return pcm

def decorate_plot(ax, dump, var, bh=True, xticks=None, yticks=None,
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


def get_xz_locations(dump, mesh=False, wrap=True, native=False, half_cut=False):
    """Get the mesh locations x_ij and z_ij needed for plotting a poloidal slice.
    By default, gets locations for phi=0,180 but can optionally just fetch half
    """
    if native:
        # We always want one "pane" when plotting in native coords
        half_cut = True
    if mesh:
        # We need a continouous set of corners representing phi=0/pi
        m = dump.grid.coord_ij_mesh(at=(0, dump['n3']//2))
        if half_cut:
            m = m[Ellipsis, 0]
        else:
            # Append reversed in th.  We're now contiguous over th=180, so we remove the last
            # (or after reversal, first) zone of the flipped (left) side
            m = np.append(m[:, :, :, 0], np.flip(m[:, :, :-1, 1], 2), 2)
    else:
        # Version for zone centers doesn't need the extra 
        m = dump.grid.coord_ij(at=(0, dump['n3']//2))
        if half_cut:
            m = m[Ellipsis, 0]
        else:
            m = np.append(m[Ellipsis, 0], np.flip(m[Ellipsis, 1], 2), 2)
    if native:
        x = m[1]
        z = m[2]
    else:
        x = dump.grid.coords.cart_x(m)
        z = dump.grid.coords.cart_z(m)

    if wrap and not mesh:
        # Wrap with first element in th, located at the first element in x/z
        x = np.append(x[Ellipsis], x[:, 0:1], 1)
        z = np.append(z[Ellipsis], z[:, 0:1], 1)

    return x, z

def get_xy_locations(dump, mesh=False, wrap=True, native=False):
    """Get the mesh locations x_ij and y_ij needed for plotting a toroidal slice."""
    if mesh:
        m = dump.grid.coord_ik_mesh()
    else:
        m = dump.grid.coord_ik()

    if native:
        x = m[1]
        y = m[3]
    else:
        x = dump.grid.coords.cart_x(m)
        y = dump.grid.coords.cart_y(m)

    if wrap and not mesh:
        # Wrap with first element in phi
        x = np.append(x[Ellipsis], x[:, 0:1], 1)
        y = np.append(y[Ellipsis], y[:, 0:1], 1)
    
    return x, y

def get_xz_var(dump, var, at=0, wrap=True, average=False, sum=False, half_cut=False):
    """Return something of the right shape to plot"""
    if average or sum:
        if isinstance(var, str):
            var = dump[var]
        var = var.mean(-1) if average else var.sum(-1)
        if not half_cut:
            var = np.append(var, np.flip(var, 1), 1)
    else:
        if isinstance(var, str):
            if half_cut:
                var = dump[:, :, at][var]
            else:
                var = np.append(dump[:, :, at][var], np.flip(dump[:, :, at + dump['n3']//2][var], 1), 1)
        else:
            if half_cut:
                var = var[:, :, at]
            else:
                var = np.append(var[:, :, at], np.flip(var[:, :, at + dump['n3']//2], 1), 1)
    if wrap:
        return np.append(var[Ellipsis], var[:, 0:1], 1)
    else:
        return var

def get_xy_var(dump, var, at=None, wrap=True, average=False, sum=False):
    if average or sum:
        if isinstance(var, str):
            var = dump[var]
        var = var.mean(1) if average else var.sum(1)
    else:
        if at is None:
            at = dump['n2']//2
        if isinstance(var, str):
            var = dump[:, at, :][var]
        else:
            var = var[:, at, :]
    if wrap:
        return np.append(var[Ellipsis], var[:, 0:1], 1)
    else:
        return var

def plot_xz(ax, dump, var, vmin=None, vmax=None, window=(-40, 40, -40, 40),
            xlabel=True, ylabel=True, native=False, log=False,
            half_cut=False, cmap='jet', shading='gouraud',
            at=0, average=False, sum=False, **kwargs):
    """Plot a poloidal or X1/X2 slice of a dump file.
    NOTE: also accepts all keyword arguments to decorate_plot()
    :param ax: Axes object to paint on
    :param dump: fluid state object
    :param vmin, vmax: colorbar minimum and maximum
    :param window: view window in X,Z coordinates, measured in r_g/c^2, 0 in BH center.
    :param xlabel, ylabel: whether to mark X/Y labels with reasonable titles
    :param native: Plot in native coordinates X1/X2 as plot X/Y axes respectively
    :param log: plot a signed quantity in logspace with symlog() above
    """

    if isinstance(var, str):
        if 'symlog_' in var:
            log = True
            var = var.replace("symlog_","")

    loc_mesh = shading == 'flat'
    x, z = get_xz_locations(dump, loc_mesh, True, native, half_cut)
    var = get_xz_var(dump, var, at, average, sum)

    if log:
        mesh = pcolormesh_symlog(ax, x, z, var, cmap=cmap, vmin=vmin, vmax=vmax,
                                 shading=shading, cbar=False) # We add a colorbar later
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
    else:
        if xlabel: ax.set_xlabel(r"x ($r_g$)")
        if ylabel: ax.set_ylabel(r"z ($r_g$)")
        if window:
            ax.set_xlim(window[:2])
            ax.set_ylim(window[2:])
        # TODO alt option of size -r_out to r_out?

        if not half_cut:
            ax.set_aspect('equal')
    
    if not 'bh' in kwargs:
        kwargs['bh'] = not native
    decorate_plot(ax, dump, var, **kwargs)

    # In case user wants to tweak this
    return mesh

def plot_xy(ax, dump, var, vmin=None, vmax=None, window=None,
            xlabel=True, ylabel=True, native=False, log=False,
            cmap='jet', shading='gouraud',
            at=None, average=False, sum=False, **kwargs):
    """Plot a toroidal or X1/X3 slice of a dump file.
    NOTE: also accepts all keyword arguments to decorate_plot()
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

    if isinstance(var, str):
        if 'symlog_' in var:
            log = True
            var = var.replace("symlog_","")

    loc_mesh = shading == 'flat'
    x, y = get_xy_locations(dump, native, loc_mesh)
    var = get_xy_var(dump, var, at, loc_mesh, average, sum)

    if log:
        mesh = pcolormesh_symlog(ax, x, y, var, cmap=cmap, vmin=vmin, vmax=vmax,
                         shading=shading, cbar=False) # We add a colorbar later
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

    if not 'bh' in kwargs:
        kwargs['bh'] = not native
    decorate_plot(ax, dump, var, **kwargs)

    # In case user wants to tweak this
    return mesh


# TODO this is currently just for profiles already in 2D
def plot_thphi(ax, dump, var, r_i, cmap='jet', vmin=None, vmax=None, window=None,
               xlabel=True, ylabel=True, native=False, log=False,
               project=False, shading='gouraud', **kwargs):
    """Plot a theta-phi slice at index r_i
    :param project 
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
    decorate_plot(ax, dump, var, bh=False, **kwargs)

    return mesh

def plot_slices(ax1, ax2, dump, var, field_overlay=True, nlines=10, **kwargs):
    """Make adjacent plots with plot_xy and plot_xz, using the given pair of axes
    """
    plot_xz(ax1, dump, var, **kwargs)
    # If we're not plotting in native coordinates, plot contours.
    # They are very unintuitive in native coords
    if field_overlay and not ('native' in kwargs.keys() and kwargs['native']):
        overlay_field(ax1, dump, nlines=nlines)

    plot_xy(ax2, dump, var, **kwargs)

def overlay_contours(ax, dump, var, levels, color='k', native=False, half_cut=False, at=0, average=True, **kwargs):
    # TODO optional line cutoff by setting NaN according to a second condition
    if isinstance(var, str):
        var = dump[var]

    x, z = get_xz_locations(dump, native=native, mesh=False, wrap=False, half_cut=half_cut)
    var = get_xz_var(dump, var, at, False, average)
    return ax.contour(x, z, var, levels=levels, colors=color, **kwargs)

def overlay_contourf(ax, dump, var, levels, color='k', native=False, half_cut=False, at=0, average=True, **kwargs):
    if isinstance(var, str):
        var = dump[var]

    x, z = get_xz_locations(dump, native=native, mesh=False, half_cut=half_cut)
    var = get_xz_var(dump, var, at, False, average)
    return ax.contourf(x, z, var, levels=levels, colors=color, **kwargs)

def overlay_field(ax, dump, **kwargs):
        overlay_flowlines(ax, dump, 'B1', 'B2', **kwargs)

def overlay_flowlines(ax, dump, varx1, varx2, nlines=20, color='k', native=False, half_cut=False, reverse=False):
    N1 = dump['n1']
    N2 = dump['n2']

    x, z = get_xz_locations(dump, native=native, half_cut=half_cut)
    varx1 = get_xz_var(dump, varx1, average=True, mesh=True, half_cut=True) * np.squeeze(dump['gdet'])
    varx2 = get_xz_var(dump, varx2, average=True, mesh=True, half_cut=True) * np.squeeze(dump['gdet'])

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
    """Overlay a quiver plot of 2 vector components onto a plot in native coordinates."""
    varx1 = get_xz_var(dump, varx1, average=True) * dump['gdet']
    varx2 = get_xz_var(dump, varx2, average=True) * dump['gdet']
    max_J = np.max(np.sqrt(varx1 ** 2 + varx2 ** 2))

    x, z = get_xz_locations(dump, native=True)

    s1 = dump['n1'] // cadence
    s2 = dump['n2'] // cadence

    ax.quiver(x[::s1, ::s2], z[::s1, ::s2], varx1[::s1, ::s2], varx2[::s1, ::s2],
              units='xy', angles='xy', scale_units='xy', scale=(cadence * max_J / norm))

# TODO Consistent idea of plane/average in x2,x3
def radial_plot(ax, dump, var, n2=0, n3=0, average=False,
                logr=False, logy=False, rlim=None, ylim=None, native=False,
                ylabel=None, title=None, **kwargs):
    r = dump['r'][:, dump['n2'] // 2, 0]
    if var.ndim == 1:
        data = var
    elif var.ndim == 2:
        data = var[:, n2]
    elif var.ndim == 3:
        if average:
            data = np.mean(var[:, n2, :], axis=-1)
        else:
            data = var[:, n2, n3]

    if native:
        ax.plot(list(range(dump['n1'])), data, **kwargs)
    else:
        ax.plot(r, data, **kwargs)

    if logr: ax.set_xscale('log')
    if logy: ax.set_yscale('log')

    if rlim: ax.set_xlim(rlim)
    if ylim: ax.set_ylim(ylim)

    ax.set_xlabel(r"$r \frac{c^2}{G M}$")
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)


def hist_2d(ax, var_x, var_y, xlbl, ylbl, title=None, logcolor=False, bins=40,
            cbar=True, cmap='jet', ticks=None):
    # Courtesy of George Wong
    var_x_flat = var_x.flatten()
    var_y_flat = var_y.flatten()
    nidx = np.isfinite(var_x_flat) & np.isfinite(var_y_flat)
    hist = np.histogram2d(var_x_flat[nidx], var_y_flat[nidx], bins=bins)
    X, Y = np.meshgrid(hist[1], hist[2])

    if logcolor:
        hist[0][hist[0] == 0] = np.min(hist[0][np.nonzero(hist[0])])
        mesh = ax.pcolormesh(X, Y, np.log10(hist[0]), cmap=cmap)
    else:
        mesh = ax.pcolormesh(X, Y, hist[0], cmap=cmap)

    # Add the patented Ben Ryan colorbar
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(mesh, cax=cax, ticks=ticks)

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)

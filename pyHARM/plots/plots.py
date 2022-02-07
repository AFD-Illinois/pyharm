################################################################################
#                                                                              #
#  UTILITIES FOR PLOTTING                                                      #
#                                                                              #
################################################################################

import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from scipy.integrate import trapz

from pyHARM.plots.pretty import pretty

"""
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

# Plotting fns: pass dump file and var as either string (key) or ndarray
# Note integrate option overrides average
# Also note label convention:
# * "known labels" are assigned true or false,
# * "unknown labels" are assigned None or a string
def plot_xz(ax, dump, var, vmin=None, vmax=None, window=(-40, 40, -40, 40),
            xlabel=True, ylabel=True, native=False, log=False,
            average=False, integrate=False, half_cut=False,
            cmap='jet', shading='flat', cbar=True, use_imshow=False,
            at=0, **kwargs):
    """Plot a slice of a dump file.
    NOTE: also accepts all keyword arguments to decorate_plot()
    :param ax: Axes object to paint on
    :param dump: fluid state object
    :param vmin, vmax: colorbar minimum and maximum
    :param window: view window in X,Z coordinates, measured in r_g/c^2, 0 in BH center.
    :param xlabel, ylabel: whether to mark X/Y labels with reasonable titles
    :param native: Plot in native coordinates X1/X2 as plot X/Y axes respectively
    :param log: plot a signed quantity in logspace with symlog() above
    :param average: phi-average of variable, rather than two slices

    """

    if isinstance(var, str):
        if 'symlog_' in var:
            log = True
            var = var.replace("symlog_","")
        var = np.append(dump[:, :, at][var], np.flip(dump[:, :, at + dump['n3']//2][var], 1), 1)

    if shading == 'flat':
        # We need a continouous set of corners representing phi=0/pi
        m = dump.grid.coord_ij_mesh(at=(0, dump['n3']//2))
        if half_cut:
            m = m[0]
        else:
            # Append reversed in th.  We're now contiguous over th=180, so we remove the last
            # (or after reversal, first) zone of the flipped side
            m = np.append(m[:, :, :, 0], np.flip(m[:, :, :-1, 1], 2), 2)
    else:
        # Version for zone centers doesn't need the extra 
        m = dump.grid.coord_ij(at=(0, dump['n3']//2))
        if half_cut:
            m = m[0]
        else:
            m = np.append(m[Ellipsis, 0], np.flip(m[Ellipsis, 1], 2), 2)
    if native:
        x = m[1]
        z = m[2]
    else:
        x = dump.grid.coords.cart_x(m)
        z = dump.grid.coords.cart_z(m)

    if shading != 'flat':
        # Wrap with first element in th, located at the first element in x/z
        var = np.append(var[Ellipsis], var[:, 0:1], 1)
        x = np.append(x[Ellipsis], x[:, 0:1], 1)
        z = np.append(z[Ellipsis], z[:, 0:1], 1)


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
            ax.set_xlim([x[0,0], x[-1,-1]])
            ax.set_ylim([z[0,0], z[-1,-1]])
    else:
        if xlabel: ax.set_xlabel(r"x ($r_g$)")
        if ylabel: ax.set_ylabel(r"z ($r_g$)")
        if window:
            ax.set_xlim(window[:2])
            ax.set_ylim(window[2:])

        if not half_cut:
            ax.set_aspect('equal')
    
    if not 'bh' in kwargs:
        kwargs['bh'] = not native

    decorate_plot(ax, dump, var, cbar=cbar, **kwargs)

def plot_xy(ax, dump, var, vmin=None, vmax=None, window=None,
            xlabel=True, ylabel=True, native=False, log=False,
            cmap='jet', shading='gouraud', cbar=True, **kwargs):

    # TODO slice for better efficiency
    if isinstance(var, str):
        if 'symlog_' in var:
            log = True
            var = var.replace("symlog_","")
        var = dump[:, dump['n2']//2, :][var]

    if shading == 'flat':
        m = dump.grid.coord_ik_mesh(at=dump['n2']//2)
    else:
        m = dump.grid.coord_ik(at=dump['n2']//2)

    if native:
        x = m[1]
        y = m[3]
    else:
        x = dump.grid.coords.cart_x(m)
        y = dump.grid.coords.cart_y(m)

    if shading != 'flat':
        # Wrap with first element in phi
        var = np.append(var[Ellipsis], var[:, 0:1], 1)
        x = np.append(x[Ellipsis], x[:, 0:1], 1)
        y = np.append(y[Ellipsis], y[:, 0:1], 1)

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
    decorate_plot(ax, dump, var, cbar=cbar, **kwargs)


# TODO this is currently just for profiles already in 2D
def plot_thphi(ax, dump, var, r_i, cmap='jet', vmin=None, vmax=None, window=None,
               label=None, xlabel=True, ylabel=True, native=False,
               project=False, shading='gouraud'):
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

    # print 'xshape is ', x.shape, ', yshape is ', y.shape, ', varshape is ', var.shape
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

# Plot two slices together without duplicating everything in the caller
def plot_slices(ax1, ax2, dump, var, field_overlay=True, nlines=10, **kwargs):
    """Make adjacent plots with plot_xy and plot_xz, on the given pair of axes.
    Basically syntactic sugar for plot_xy and plot_xz.
    """
    if 'arrspace' in list(kwargs.keys()):
        arrspace = kwargs['arrspace']
    else:
        arrspace = False

    plot_xz(ax1, dump, var, **kwargs)
    if field_overlay:
        overlay_field(ax1, dump, nlines=nlines, native=arrspace)

    plot_xy(ax2, dump, var, **kwargs)

def overlay_contours(ax, dump, var, levels, color='k', **kwargs):
    # TODO optional line cut by setting NaN according to second condition
    if isinstance(var, str):
        var = dump[var]

    x = _flatten_xz(dump['x'])
    z = _flatten_xz(dump['z'])
    var = _flatten_xz(var, average=True)
    return ax.contour(x, z, var, levels=levels, colors=color, **kwargs)

def overlay_contourf(ax, dump, var, levels, color='k', **kwargs):
    if isinstance(var, str):
        var = dump[var]

    x = _flatten_xz(dump['x'])
    z = _flatten_xz(dump['z'])
    var = _flatten_xz(var, average=True)
    return ax.contourf(x, z, var, levels=levels, colors=color, **kwargs)

def overlay_field(ax, dump, native=False, nlines=20, **kwargs):
    if not native:
        overlay_flowlines(ax, dump, dump['B1'], dump['B2'], nlines=nlines, **kwargs)

def overlay_flowlines(ax, dump, varx1, varx2, nlines=50, reverse=False):
    N1 = dump['n1']
    N2 = dump['n2']

    x = _flatten_xz(dump['x'])
    z = _flatten_xz(dump['z'])

    varx1 = varx1.mean(axis=-1)
    varx2 = varx2.mean(axis=-1)
    AJ_phi = np.zeros([2 * N1, N2])
    gdet = dump['gdet']
    for j in range(N2):
        for i in range(N1):
            if not reverse:
                AJ_phi[N1 - 1 - i, j] = AJ_phi[i + N1, j] = (
                        trapz(gdet[:i, j] * varx2[:i, j], dx=dump['dx1']) -
                        trapz(gdet[i, :j] * varx1[i, :j], dx=dump['dx2']))
            else:
                AJ_phi[N1 - 1 - i, j] = AJ_phi[i + N1, j] = (
                        trapz(gdet[:i, j] * varx2[:i, j], dx=dump['dx1']) +
                        trapz(gdet[i, j:] * varx1[i, j:], dx=dump['dx2']))
    AJ_phi -= AJ_phi.min()
    levels = np.linspace(0, AJ_phi.max(), nlines * 2)

    ax.contour(x, z, AJ_phi, levels=levels, colors='k')


def overlay_quiver(ax, dump, varx1, varx2, cadence=64, norm=1):
    varx1 *= dump['gdet']
    varx2 *= dump['gdet']
    max_J = np.max(np.sqrt(varx1 ** 2 + varx2 ** 2))
    x1_norm = (dump['X1'] - dump['startx1']) / (dump['n1'] * dump['dx1'])
    x2_norm = (dump['X2'] - dump['startx2']) / (dump['n2'] * dump['dx2'])
    x = _flatten_xz(x1_norm)[dump['n1']:, :]
    z = _flatten_xz(x2_norm)[dump['n1']:, :]

    s1 = dump['n1'] // cadence
    s2 = dump['n2'] // cadence

    ax.quiver(x[::s1, ::s2], z[::s1, ::s2], varx1[::s1, ::s2], varx2[::s1, ::s2],
              units='xy', angles='xy', scale_units='xy', scale=cadence * max_J / norm)

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


def _flatten_xz(array, at=0, average=False, patch_pole=False):
    """Get an XZ (vertical) slice or average of 3D polar data array[nr,nth,nphi]
    Returns an array of size (2*N1, N2) containing the phi={0,pi} slice,
    i.e. both "wings" of data around the BH at center
    :param patch_pole: set the first & last rows to 0 -- used when flattening arrays
    of x-coordinates to avoid a visual artifact
    :param average: whether to average instead of slicing
    """
    # TODO get phi != 0 slices too
    if array.ndim == 3:
        N1 = array.shape[0]
        N2 = array.shape[1]
        N3 = array.shape[2]
        flat = np.zeros([2 * N1, N2])
        if average:
            for i in range(N1):
                # Produce identical hemispheres to get the right size output
                # TODO option to average halves?
                flat[i, :] = np.mean(array[N1 - 1 - i, :, :], axis=-1)
                flat[i + N1, :] = np.mean(array[i, :, :], axis=-1)
        else:
            for i in range(N1):
                flat[i, :] = array[N1 - 1 - i, :, at + N3 // 2]
                flat[i + N1, :] = array[i, :, at]

    elif array.ndim == 2:
        N1 = array.shape[0]
        N2 = array.shape[1]
        flat = np.zeros([2 * N1, N2])
        for i in range(N1):
            flat[i, :] = array[N1 - 1 - i, :]
            flat[i + N1, :] = array[i, :]

    # Theta is technically [small,pi/2-small]
    # This patches the X coord so the plot looks nice
    if patch_pole:
        flat[:, 0] = 0
        flat[:, -1] = 0

    return flat

def _flatten_12(array, average=False):
    """Get a single 2D slice or average of 3D data
    Currently limited to x3=0 slice
    :param average: whether to average instead of slicing
    """
    # TODO get phi != 0 slices too
    if array.ndim == 3:
        if average:
            slice = np.mean(array, axis=-1)
        else:
            slice = array[:, :, 0]
    elif array.ndim == 2:
        slice = array

    return slice

# Get xy slice of 3D data
def _flatten_xy(array, average=False, loop=True):
    if array.ndim == 3:
        if average:
            slice = np.mean(array, axis=1)
        else:
            slice = array[:, array.shape[1] // 2, :]
    elif array.ndim == 2:
        slice = array

    if loop:
        return _loop_phi(slice)
    else:
        return slice

def _flatten_yz(array, at_i, average=False, loop=True):
    if array.ndim == 3:
        if average:
            slice = np.mean(array, axis=0)
        else:
            slice = array[at_i, :, :]
    elif array.ndim == 2:
        slice = array

    if loop:
        return _loop_phi(slice)
    else:
        return slice

def _loop_phi(array):
    return np.vstack((array.transpose(), array.transpose()[0])).transpose()

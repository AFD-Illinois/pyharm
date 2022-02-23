import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""Generic utilities and plot types -- anything potentially useful/stolen from outside pyharm
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
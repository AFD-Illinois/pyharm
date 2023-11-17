__license__ = """
 File: plot_utils.py
 
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

import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

__doc__ = """Generic utilities and plot types -- anything plotting-related which is
potentially useful (or indeed stolen from) outside pyharm
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
        # Allow specifying vmin/vmax as if everything was positive
        if vmin is not None and vmin*vmax > 0:
            decades = np.log10(vmax/vmin)
        vmin = -vmax

    int_min_pow, int_max_pow = int(np.ceil(np.log10(-vmin))), int(np.ceil(np.log10(vmax)))

    if linthresh is None:
        linthresh = vmax * 10**(-decades)

    logthresh = int(np.ceil(np.log10(linthresh)))
    tick_locations = ([vmin]
                      + [-(10.0 ** x) for x in range(int_min_pow - 1, logthresh - 1, -1)]
                      + [0.0]
                      + [(10.0 ** x) for x in range(logthresh, int_max_pow)]
                      + [vmax])
    pcm = ax.pcolormesh(X, Y, Z, norm=colors.SymLogNorm(linthresh=linthresh, linscale=linscale, base=10, vmin=-vmax, vmax=vmax),
                         cmap=cmap, **kwargs)
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pcm, cax=cax, ticks=tick_locations, format=ticker.LogFormatterMathtext())

    return pcm

def pcolormesh_log(ax, X, Y, Z, vmax=None, vmin=None, cmap='jet', cbar=True, **kwargs):
    """Wrapper for matplotlib's pcolormesh that uses it sensibly, instead of the defaults.

    If not specified, vmax is set automatically
    In order to keep colors sensible, vmin is overridden unless set alone.
    """

    pcm = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                         cmap=cmap, **kwargs)
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pcm, cax=cax, format=ticker.LogFormatterMathtext())

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
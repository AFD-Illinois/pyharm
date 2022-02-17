
import os, sys
import click

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyHARM
from pyHARM import io
from pyHARM.plots import figures, plots
from pyHARM.plots.pretty import pretty


def frame(fname, diag, kwargs):
    tstart, tend = kwargs['tstart'], kwargs['tend']
    movie_type = kwargs['movie_type']
    tdump = io.get_dump_time(fname)
    frame_name = os.path.join("frames_"+movie_type, "frame_t%08d.png" % int(tdump))
    if (tstart is not None and tdump < tstart) or \
        (tend is not None and tdump > tend) or \
        ('resume' in kwargs and kwargs['resume'] and os.path.exists(frame_name)):
        return

    print("Imaging t={}".format(int(tdump)), file=sys.stderr)

    dump = pyHARM.load_dump(fname)

    # Zoom in for small problems
    if len(dump['r'].shape) == 1:
        sz = 50
        nlines = 20
        rho_l, rho_h = None, None
    elif len(dump['r'].shape) == 2:
        sz = 200
        nlines = 10
        rho_l, rho_h = -6, 1
    else:
        if dump['r'][-1, 0, 0] > 100:
            sz = 200
            nlines = 10
            rho_l, rho_h = -6, 1
        elif dump['r'][-1, 0, 0] > 10:
            sz = 50
            nlines = 5
            rho_l, rho_h = -6, 1
        else: # Then this is a Minkowski simulation or something weird. Guess.
            sz = (dump['x'][-1,0,0] - dump['x'][0,0,0]) / 2
            nlines = 0
            rho_l, rho_h = -2, 0.0

    if kwargs['size'] is not None:
        sz = float(kwargs['size'])

    window = [-sz, sz, -sz, sz]

    if "_array" in movie_type:
        native_coords = True
        window = None # Let plotter choose based on grid
    else:
        native_coords = False

    figx, figy = kwargs['fig_x'], kwargs['fig_y']

    if movie_type in figures.__dict__:
        fig = figures.__dict__[movie_type]()

    else:
        fig = plt.figure(figsize=(figx, figy))
        # Strip global flags from the movie string
        l_movie_type = movie_type
        if "_ghost" in movie_type:
            l_movie_type = l_movie_type.replace("_ghost","")
        if "_array" in l_movie_type:
            l_movie_type = l_movie_type.replace("_array","")
        at = 0
        if "_cross" in l_movie_type:
            l_movie_type = l_movie_type.replace("_cross","")
            at = dump['n3']//2
        if "_quarter" in l_movie_type:
            l_movie_type = l_movie_type.replace("_quarter","")
            at = dump['n3']//4
        
        shading = kwargs['shading']

        # Try to make a simple movie of just the stated variable
        if "_poloidal" in l_movie_type or "_2d" in l_movie_type:
            ax = plt.subplot(1, 1, 1)
            var = l_movie_type.replace("_poloidal","")
            plots.plot_xz(ax, dump, var, at=at, label=pretty(var),
                        vmin=None, vmax=None, window=window, native=native_coords,
                        xlabel=False, ylabel=False, xticks=[], yticks=[],
                        cbar=True, cmap='jet', field_overlay=False, shading=shading)
        elif "_toroidal" in l_movie_type:
            ax = plt.subplot(1, 1, 1)
            var = l_movie_type.replace("_toroidal","")
            plots.plot_xy(ax, dump, var, at=at, label=pretty(var),
                        vmin=rho_l, vmax=rho_h, window=window, native=native_coords,
                        cbar=True, cmap='jet', shading=shading)
        elif "_1d" in l_movie_type:
            ax = plt.subplot(1, 1, 1)
            var = l_movie_type.replace("_1d","")
            ax.plot(dump['x'], dump[var][:,0,0], label=pretty(var))
            ax.set_ylim((rho_l, rho_h))
            ax.set_title(pretty(var))
        else:
            ax_slc = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
            ax = ax_slc[0]
            var = l_movie_type
            plots.plot_slices(ax_slc[0], ax_slc[1], dump, var, at=at, label=pretty(l_movie_type),
                        vmin=rho_l, vmax=rho_h, window=window, native=native_coords,
                        cbar=True, cmap='jet', field_overlay=False, shading=shading)

        # Labels
        if "divB" in movie_type:
            fig.suptitle(r"Max $\nabla \cdot B$ = {}".format(np.max(np.abs(dump['divB']))))

        if "jsq" in movie_type:
            fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)

        # if not native_coords:
        #    plots.overlay_field(ax, dump, nlines=nlines)

    # Title by time, otherwise number
    fig.suptitle("t = {}".format(int(tdump)))
    plt.subplots_adjust(left=0.03, right=0.97)
    plt.savefig(frame_name, dpi=kwargs['fig_dpi'])

    return fname
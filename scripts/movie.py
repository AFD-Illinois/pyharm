#!/usr/bin/env python3
# Make a movie (folder full of images representing frames) with the output from a simulation

# Usage: movie.py [type] [/path/to/dumpfiles]

# Where [type] is a string passed to the function below representing what plotting to do in each frame,
# and [/path/to/dumpfiles] is the path to the *folder* containing HDF5 output in any form which pyHARM can read

# Generally good overview movies are 'simplest' & 'traditional', see the function body for details.

import os
import click
import psutil
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pyHARM
import pyHARM.io as io
import pyHARM.io.iharm3d_header as hdr
from pyHARM.reductions import *

import pyHARM.plots.figures as figures
import pyHARM.plots.plots as pplt
import pyHARM.plots.plot_results as ppltr
from pyHARM.plots.pretty import pretty
from pyHARM.util import i_of, calc_nthreads, run_parallel

files = []
frame_dir = ""

@click.command()
@click.argument('movie_type')
@click.argument('path')
# Common options
@click.option('-s', '--tstart', default=0, help="Start time.")
@click.option('-e', '--tend', default=1e7, help="End time.")
# If you're invoking manually
@click.option('--fig_x', default=16, help="Figure width in inches.")
@click.option('--fig_y', default=9, help="Figure height in inches.")
@click.option('--fig_dpi', default=100, help="DPI of resulting figure.")
@click.option('-g','--plot_ghost', is_flag=True, default=False, help="Plot ghost zones.")
@click.option('-sz','--size', default=None, help="Window size, in M each side of central object.")
@click.option('-sh','--shading', default='gouraud', help="Shading: flat, nearest, gouraud.")
# Extras
@click.option('-r', '--resume', is_flag=True, default=False, help="Continue a previous run, by skipping existing frames")
@click.option('-d', '--debug', is_flag=True, default=False, help="Serial operation for debugging")
@click.option('-m', '--memory_limit', is_flag=True, default=False,
                help="Limit parallel operations to memory instead of processor count, for large files.")
# TODO reimplement this
@click.option('--diag_post', is_flag=True, default=False, help="End time.")

def movie(movie_type, path, debug, memory_limit, diag_post, tstart, tend,
          fig_x, fig_y, size, shading, fig_dpi, plot_ghost, resume):
    """Movie of type MOVIE_TYPE from dumps at PATH.
    Generate the frames of a movie running over all dumps at the given path.
    """
    # Try to load known filenames
    global files
    files = io.get_fnames(path)

    global frame_dir
    frame_dir = "frames_" + movie_type
    os.makedirs(frame_dir, exist_ok=True)

    # TODO diag loading.  Dynamic based on movie type?
    if diag_post:
        # Load fluxes from post-analysis: more flexible
        #diag = io.load_results()
        pass
    else:
        # Load diagnostics from HARM itself
        #diag = io.load_log(path)
        pass

    if debug:
        # Run sequentially to make backtraces work
        for i in range(len(files)):
            frame(i)
    else:
        if memory_limit:
            nthreads = calc_nthreads(io.read_hdr(files[0]), pad=0.6)
            if psutil.cpu_count() < nthreads:
                nthreads = psutil.cpu_count()
        else:
            nthreads = psutil.cpu_count()
            print("Using {} processes".format(nthreads))

        run_parallel(frame, len(files), nthreads)


@click.pass_context
def frame(ctx, n):
    tstart, tend = ctx.params['tstart'], ctx.params['tend']
    movie_type = ctx.params['movie_type']
    tdump = io.get_dump_time(files[n])
    if (tstart is not None and tdump < tstart) or \
        (tend is not None and tdump > tend):
        return

    print("frame {} / {}".format(n, len(files)-1))

    dump = pyHARM.load_dump(files[n])

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

    if ctx.params['size'] is not None:
        sz = float(ctx.params['size'])

    window = [-sz, sz, -sz, sz]

    # If we're in arrspace we (almost) definitely want a 0,1 window
    if "_array" in movie_type:
        native_coords = True
        window = None # Let plotter choose based on grid
    else:
        native_coords = False

    figx, figy = ctx.params['fig_x'], ctx.params['fig_y']

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
        
        shading = ctx.params['shading']

        # Try to make a simple movie of just the stated variable
        if "_poloidal" in l_movie_type:
            ax = plt.subplot(1, 1, 1)
            var = l_movie_type.replace("_poloidal","")
            pplt.plot_xz(ax, dump, var, at=at, label=pretty(var),
                        vmin=None, vmax=None, window=window, native=native_coords,
                        xlabel=False, ylabel=False, xticks=[], yticks=[],
                        cbar=True, cmap='jet', field_overlay=False, shading=shading)
        elif "_toroidal" in l_movie_type:
            ax = plt.subplot(1, 1, 1)
            var = l_movie_type.replace("_toroidal","")
            pplt.plot_xy(ax, dump, var, at=at, label=pretty(var),
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
            pplt.plot_slices(ax_slc[0], ax_slc[1], dump, var, at=at, label=pretty(l_movie_type),
                        vmin=rho_l, vmax=rho_h, window=window, native=native_coords,
                        cbar=True, cmap='jet', field_overlay=False, shading=shading)

        # Labels
        if "divB" in movie_type:
            plt.suptitle(r"Max $\nabla \cdot B$ = {}".format(np.max(np.abs(dump['divB']))))

        if "jsq" in movie_type:
            plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)

        #if not native_coords:
    	#    pplt.overlay_field(ax, dump, nlines=nlines)

    # Title by time, otherwise number
    try:
       fig.suptitle("t = {}".format(int(dump['t'])))
    except ValueError:
       fig.suptitle("dump {}".format(n))
    plt.subplots_adjust(left=0.03, right=0.97)
    plt.savefig(os.path.join(frame_dir, 'frame_%08d.png' % n), dpi=ctx.params['fig_dpi'])
    plt.close(fig)

if __name__ == "__main__":
    movie()
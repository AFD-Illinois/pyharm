#!/usr/bin/env python3
# Make a movie (folder full of images representing frames) with the output from a simulation

# Usage: movie.py [type] [/path/to/dumpfiles]

# Where [type] is a string passed to the function below representing what plotting to do in each frame,
# and [/path/to/dumpfiles] is the path to the *folder* containing HDF5 output in any form which pyHARM can read

# Generally good overview movies are 'simplest' & 'traditional', see the function body for details.


import os
import sys
import psutil
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pyHARM
import pyHARM.io as io
from pyHARM.ana.reductions import *
import pyHARM.ana.plot as pplt
import pyHARM.ana.plot_results as ppltr
from pyHARM.ana.variables import T_mixed, pretty
from pyHARM.util import i_of, calc_nthreads, run_parallel

# Movie size in inches. Keep 16/9 for standard-size movies
FIGX = 1
FIGY = 1
FIGDPI = 8192

# Scale
rho_l = -6
rho_h = 1

# Default movie start & end time.
# Can be overridden on command line for splitting movies among processes
tstart = 0
tend = 1e7

def plot(n):
    tdump = io.get_dump_time(files[n])
    if (tstart is not None and tdump < tstart) or (tend is not None and tdump > tend):
        return
    
    print("frame {} / {}".format(n, len(files)-1))
    
    fig = plt.figure(figsize=(FIGX, FIGY))
    
    to_load = {}
    to_load['add_grid_caches'] = False
    to_load['calc_derived'] = False
    if "current" in movie_type or "jsq" in movie_type or "jcon" in movie_type:
        to_load['add_jcon'] = True
    if "divB" in movie_type:
        to_load['add_divB'] = True
        #to_load['calc_divB'] = True
    if "psi_cd" in movie_type:
        to_load['add_psi_cd'] = True
    if "1d" in movie_type:
        to_load['add_grid_caches'] = False
        to_load['calc_derived'] = False
    if "_ghost" in movie_type:
        plot_ghost = True
        to_load['add_ghosts'] = True
    else:
        plot_ghost = False
    # TODO U if needed

    dump = pyHARM.load_dump(files[n], **to_load)

    USEARRSPACE = True
    if plot_ghost:
        window = [-0.1, 1.1, -0.1, 1.1]
    else:
        window = [0, 1, 0, 1]
    
    if movie_type == "simplest_poloidal":
        # Simplest movie: just RHO, poloidal slice
        ax_slc = plt.subplot(1, 1, 1)
        var = 'rho'
        arrspace=False
        vmin = None
        vmax = None
        pplt.plot_xz(ax_slc, dump, var, label="",
                     vmin=vmin, vmax=vmax, window=window, arrayspace=arrspace,
                     xlabel=False, ylabel=False, xticks=[], yticks=[],
                     cbar=False, cmap='jet', use_imshow=True)
        ax_slc.axis('off')
        plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)

    elif movie_type == "prims_poloidal":
        ax_slc = lambda i: plt.subplot(2, 4, i)
        vmin, vmax = None, None
        pplt.plot_xz(ax_slc(1), dump, 'RHO', label="",
                     vmin=vmin, vmax=vmax, window=window, arrayspace=True,
                     xlabel=False, ylabel=False, xticks=[], yticks=[],
                     cbar=False, cmap='jet')
        pplt.plot_xz(ax_slc(2), dump, 'UU', label="",
                     vmin=vmin, vmax=vmax, window=window, arrayspace=True,
                     xlabel=False, ylabel=False, xticks=[], yticks=[],
                     cbar=False, cmap='jet')
        pplt.plot_xz(ax_slc(3), dump, 'U1', label="",
                     vmin=vmin, vmax=vmax, window=window, arrayspace=True,
                     xlabel=False, ylabel=False, xticks=[], yticks=[],
                     cbar=False, cmap='jet')
        pplt.plot_xz(ax_slc(4), dump, 'U2', label="",
                     vmin=vmin, vmax=vmax, window=window, arrayspace=True,
                     xlabel=False, ylabel=False, xticks=[], yticks=[],
                     cbar=False, cmap='jet')
        pplt.plot_xz(ax_slc(5), dump, 'U3', label="",
                     vmin=vmin, vmax=vmax, window=window, arrayspace=True,
                     xlabel=False, ylabel=False, xticks=[], yticks=[],
                     cbar=False, cmap='jet')
        pplt.plot_xz(ax_slc(6), dump, 'B1', label="",
                     vmin=vmin, vmax=vmax, window=window, arrayspace=True,
                     xlabel=False, ylabel=False, xticks=[], yticks=[],
                     cbar=False, cmap='jet')
        pplt.plot_xz(ax_slc(7), dump, 'B2', label="",
                     vmin=vmin, vmax=vmax, window=window, arrayspace=True,
                     xlabel=False, ylabel=False, xticks=[], yticks=[],
                     cbar=False, cmap='jet')
        pplt.plot_xz(ax_slc(8), dump, 'B3', label="",
                     vmin=vmin, vmax=vmax, window=window, arrayspace=True,
                     xlabel=False, ylabel=False, xticks=[], yticks=[],
                     cbar=False, cmap='jet')

    elif movie_type == "vectors":
        ax_slc = lambda i: plt.subplot(2, 4, i)
        # Usual movie: RHO beta fluxes
        # CUTS
        pplt.plot_slices(ax_slc(1), ax_slc(5), dump, 'log_rho', label=pretty('log_rho'), average=True,
                        vmin=rho_l, vmax=rho_h, cmap='jet', window=window, arrayspace=USEARRSPACE)

        for i,var in zip((2,3,4,6,7,8), ("U1", "U2", "U3", "B1", "B2", "B3")):
            pplt.plot_xz(ax_slc(i), dump, np.log10(dump[var]), label=var,
                            vmin=rho_l, vmax=rho_h, cmap='Reds', window=window, arrayspace=USEARRSPACE)
            pplt.plot_xz(ax_slc(i), dump, np.log10(-dump[var]), label=var,
                            vmin=rho_l, vmax=rho_h, cmap='Blues', window=window, arrayspace=USEARRSPACE)

    elif movie_type == "vecs_cov":
        ax_slc = lambda i: plt.subplot(2, 4, i)
        for i,var in zip((1,2,3,4,5,6,7,8), ("u_0", "u_r", "u_th", "u_3","b_0", "b_r", "b_th", "b_3")):
            pplt.plot_xz(ax_slc(i), dump, np.log10(dump[var]), label=var,
                            vmin=rho_l, vmax=rho_h, cmap='Reds', window=window, arrayspace=USEARRSPACE)
            pplt.plot_xz(ax_slc(i), dump, np.log10(-dump[var]), label=var,
                            vmin=rho_l, vmax=rho_h, cmap='Blues', window=window, arrayspace=USEARRSPACE)

    elif movie_type == "vecs_con":
        ax_slc = lambda i: plt.subplot(2, 4, i)
        for i,var in zip((1,2,3,4,5,6,7,8), ("u^0", "u^r", "u^th", "u^3","b^0", "b^r", "b^th", "b^3")):
            pplt.plot_xz(ax_slc(i), dump, np.log10(dump[var]), label=var,
                            vmin=rho_l, vmax=rho_h, cmap='Reds', window=window, arrayspace=USEARRSPACE)
            pplt.plot_xz(ax_slc(i), dump, np.log10(-dump[var]), label=var,
                            vmin=rho_l, vmax=rho_h, cmap='Blues', window=window, arrayspace=USEARRSPACE)

    elif movie_type == "ejection":
        ax_slc = lambda i: plt.subplot(1, 2, i)
        # Usual movie: RHO beta fluxes
        # CUTS
        pplt.plot_xz(ax_slc(1), dump, 'log_rho', label=pretty('log_rho')+" phi-average", average=True,
                        vmin=rho_l, vmax=rho_h, cmap='jet', window=window, arrayspace=USEARRSPACE)
        pplt.plot_xz(ax_slc(2), dump, 'log_bsq', label=pretty('log_bsq')+" phi-average", average=True,
                        vmin=rho_l, vmax=rho_h, cmap='jet', window=window, arrayspace=USEARRSPACE)

    elif movie_type == "b_bug":
        rmax = 10
        thmax = 10
        phi = 100
        ax_slc = lambda i: plt.subplot(1, 3, i)
        ax_slc(1).pcolormesh(dump['X1'][:rmax,0:thmax,phi], dump['X2'][:rmax,0:thmax,phi], dump['log_b^r'][:rmax,0:thmax,phi], vmax=0, vmin=-4)
        ax_slc(2).pcolormesh(dump['X1'][:rmax,0:thmax,phi], dump['X2'][:rmax,0:thmax,phi], dump['log_b^th'][:rmax,0:thmax,phi], vmax=0, vmin=-4)
        ax_slc(3).pcolormesh(dump['X1'][:rmax,0:thmax,phi], dump['X2'][:rmax,0:thmax,phi], dump['log_b^3'][:rmax,0:thmax,phi], vmax=0, vmin=-4)

    elif movie_type == "floors":
        ax_slc = lambda i: plt.subplot(2, 4, i)
        pplt.plot_xz(ax_slc(1), dump, 'log_rho', label=pretty('log_rho'),
                        vmin=rho_l, vmax=rho_h, cmap='jet', window=window, arrayspace=USEARRSPACE)
        max_fail = 20
        pplt.plot_xz(ax_slc(2), dump, dump['floors'] & 1, label="GEOM_RHO",
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, arrayspace=USEARRSPACE)
        pplt.plot_xz(ax_slc(3), dump, dump['floors'] & 2, label="GEOM_U",
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, arrayspace=USEARRSPACE)
        pplt.plot_xz(ax_slc(4), dump, dump['floors'] & 4, label="B_RHO",
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, arrayspace=USEARRSPACE)
        pplt.plot_xz(ax_slc(5), dump, dump['floors'] & 8, label="B_U",
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, arrayspace=USEARRSPACE)
        pplt.plot_xz(ax_slc(6), dump, dump['floors'] & 16, label="TEMP",
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, arrayspace=USEARRSPACE)
        pplt.plot_xz(ax_slc(7), dump, dump['floors'] & 32, label="GAMMA",
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, arrayspace=USEARRSPACE)
        pplt.plot_xz(ax_slc(8), dump, dump['floors'] & 64, label="KTOT",
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, arrayspace=USEARRSPACE)

    else:
        # Strip global flags from the movie string
        l_movie_type = movie_type
        if "_ghost" in movie_type:
            l_movie_type = l_movie_type.replace("_ghost","")
        if "_array" in l_movie_type:
            l_movie_type = l_movie_type.replace("_array","")
        at = 0
        if "_cross" in l_movie_type:
            l_movie_type = l_movie_type.replace("_cross","")
            at = dump['n2']//2

        # Try to make a simple movie of just the stated variable
        # These are *informal*.  Renormalize the colorscheme however we want
        #rho_l, rho_h = None, None
        if "_poloidal" in l_movie_type:
            ax = plt.subplot(1, 1, 1)
            var = l_movie_type.replace("_poloidal","")
            pplt.plot_xz(ax, dump, var, at=at, label=pretty(var),
                        vmin=rho_l, vmax=rho_h, window=window, arrayspace=USEARRSPACE,
                        xlabel=False, ylabel=False, xticks=[], yticks=[],
                        cbar=False, cmap='jet', field_overlay=False, shading=('gouraud', 'flat')[USEARRSPACE])
        elif "_toroidal" in l_movie_type:
            ax = plt.subplot(1, 1, 1)
            var = l_movie_type.replace("_toroidal","")
            pplt.plot_xy(ax, dump, var, at=at, label=pretty(var),
                        vmin=rho_l, vmax=rho_h, window=window, arrayspace=USEARRSPACE,
                        cbar=True, cmap='jet', shading=('gouraud', 'flat')[USEARRSPACE])
        elif "_1d" in l_movie_type:
            ax = plt.subplot(1, 1, 1)
            var = l_movie_type.replace("_1d","")
            ax.plot(dump['x'], dump[var][:,0,0], label=pretty(var))
            ax.set_ylim((rho_l, rho_h))
            ax.set_title(pretty(var))
        else:
            ax_slc = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
            var = l_movie_type
            pplt.plot_slices(ax_slc[0], ax_slc[1], dump, var, at=at, label=pretty(l_movie_type),
                        vmin=rho_l, vmax=rho_h, window=window, arrayspace=USEARRSPACE,
                        cbar=True, cmap='jet', field_overlay=False, shading=('gouraud', 'flat')[USEARRSPACE])
        
        # Labels
        if "divB" in movie_type:
            plt.suptitle(r"Max $\nabla \cdot B$ = {}".format(np.max(np.abs(dump['divB']))))

        if "jsq" in movie_type:
            plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)

    plt.subplots_adjust(left=0.0, right=0.0, top=0.0, bottom=0.0)
    plt.savefig(os.path.join(frame_dir, 'frame_%08d.png' % n), dpi=FIGDPI)
    plt.close(fig)

    del dump


if __name__ == "__main__":
    # Process arguments
    movie_type = sys.argv[1]
    path = sys.argv[2]
    if len(sys.argv) > 3:
        tstart = float(sys.argv[3])
    if len(sys.argv) > 4:
        tend = float(sys.argv[4])

    # Try to load known filenames
    files = io.get_fnames(path)

    frame_dir = "frames_" + movie_type
    os.makedirs(frame_dir, exist_ok=True)
    
    for i in range(len(files)):
        plot(i)

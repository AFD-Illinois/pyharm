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
FIGX = 16
FIGY = 9
FIGDPI = 100

# Load diagnostic data from post-processing (eht_out.p)
diag_post = False

# Default movie start & end time.
# Can be overridden on command line for splitting movies among processes
tstart = 0
tend = 1e7

def plot(n):
    tdump = io.get_dump_time(files[0][n])
    if (tstart is not None and tdump < tstart) or (tend is not None and tdump > tend):
        return
    
    print("frame {} / {}".format(n, len(files[0])-1))
    
    fig, ax_array = plt.subplots(2, 5, figsize=(FIGX, FIGY))
    ax = fig.axes
    
    to_load = {}
    if "simple" not in movie_type and "floor" not in movie_type and "rho" not in movie_type:
        # Everything but simple & pure floor movies needs derived vars
        to_load['calc_derived'] = True
    if "simple" in movie_type:
        # Save memory
        #to_load['add_grid_caches'] = False
        pass
    if "fail" in movie_type or "e_ratio" in movie_type or "conservation" in movie_type:
        to_load['add_fails'] = True
    if "floor" in movie_type:
        to_load['add_floors'] = True
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

    dumps = []
    for i in range(len(files)):
        dumps.append(pyHARM.load_dump(files[i][n], **to_load))

    # Title by time, otherwise number
    try:
        fig.suptitle("t = {}".format(int(dumps[0]['t'])))
    except ValueError:
        fig.suptitle("dump {}".format(n))

    rho_l, rho_h = -5, 1.5
    window = [-40, 0, 0, 80]
    
    # Strip global flags from the movie string
    l_movie_type = movie_type
    if "_ghost" in movie_type:
        l_movie_type = l_movie_type.replace("_ghost","")
    if "_array" in l_movie_type:
        l_movie_type = l_movie_type.replace("_array","")
    at = 0
    if "_cross" in l_movie_type:
        l_movie_type = l_movie_type.replace("_cross","")
        at = dumps[0]['n2']//2
    if "_avg" in l_movie_type:
        l_movie_type = l_movie_type.replace("_avg","")
        do_average = True
    else:
        do_average = False
    use_arrspace = False

    for i,dump in enumerate(dumps):
        # Try to make simple plots of just the stated variable
        # These are *informal*.  Renormalize the colorscheme however we want
        #rho_l, rho_h = None, None
        if "_poloidal" in l_movie_type:
            var = l_movie_type.replace("_poloidal","")
            pplt.plot_xz(ax[i], dump, var, at=at, label=pretty(var),
                        vmin=rho_l, vmax=rho_h, window=window, arrayspace=use_arrspace, average=do_average,
                        xlabel=False, ylabel=False, title=False, #xticks=[], yticks=[],
                        cbar=(i % 5 == 4), cmap='jet', field_overlay=True, shading=('gouraud', 'flat')[use_arrspace])
        elif "_toroidal" in l_movie_type:
            var = l_movie_type.replace("_toroidal","")
            pplt.plot_xy(ax[i], dump, var, at=at, label=pretty(var),
                        vmin=rho_l, vmax=rho_h, window=window, arrayspace=use_arrspace, average=do_average,
                        title=False,
                        cbar=True, cmap='jet', shading=('gouraud', 'flat')[use_arrspace])
        elif "_1d" in l_movie_type:
            var = l_movie_type.replace("_1d","")
            ax[i].plot(dump['x'], dump[var][:,0,0], label=pretty(var))
            ax[i].set_ylim((rho_l, rho_h))
        else:
            raise NotImplementedError
        
        # Labels
        if "divB" in movie_type:
            plt.suptitle(r"Max $\nabla \cdot B$ = {}".format(np.max(np.abs(dump['divB']))))

        if "jsq" in movie_type:
            plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)

    # if not use_arrspace:
    # 	pplt.overlay_contours(ax, dump, 'sigma', [1])

    #plt.subplots_adjust(left=0.03, right=0.97)
    plt.tight_layout()
    plt.savefig(os.path.join(frame_dir, 'frame_%08d.png' % n), dpi=FIGDPI)
    plt.close(fig)

    del dump


if __name__ == "__main__":
    # Process arguments
    if sys.argv[1] == '-d':
        debug = True
        movie_type = sys.argv[2]
        path = sys.argv[3]
        if len(sys.argv) > 4:
            tstart = float(sys.argv[4])
        if len(sys.argv) > 5:
            tend = float(sys.argv[5])
    else:
        debug = False
        movie_type = sys.argv[1]
        tstart = float(sys.argv[2])
        tend = float(sys.argv[3])
        paths = sys.argv[4:]


    # Try to load known filenames
    files = []
    for path in paths:
        files.append(io.get_fnames(path))

    frame_dir = "frames_" + movie_type
    os.makedirs(frame_dir, exist_ok=True)
    
    if debug:
        # Run sequentially to make backtraces work
        for i in range(len(files)):
            plot(i)
    else:
        nthreads = 10
        run_parallel(plot, len(files[0]), nthreads)

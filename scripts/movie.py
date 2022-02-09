#!/usr/bin/env python3

"""
Make a plot of some variable over a simulation domain, either one image of a single dump file,
or a movie (folder of image "frames") from many dump files.

If this script can import MPI and sees an MPI.COMM_WORLD, it attempts to use it.
Otherwise, it uses multiprocessing, defaulting to as many processes as a node has cores.
"""

import os, sys
import click
import glob
import psutil
import multiprocessing

try:
    import mpi4py
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor, MPIPoolExecutor
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Using MPI", file=sys.stderr)
    use_mpi = True
except ImportError:
    use_mpi = False

import numpy as np
import pyHARM.io as io
from pyHARM.plots.frame import frame
from pyHARM.util import calc_nthreads

@click.command()
@click.argument('movie_type')
@click.argument('path', type=click.Path(exists=True))
# Common options
@click.option('-s', '--tstart', default=0, help="Start time.")
@click.option('-e', '--tend', default=1e7, help="End time.")
@click.option('-n', '--nthreads', default=None, help="Number of processes to use, across all nodes")
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
@click.option('-m', '--memory_limit', default=1, help="Memory limit in GB for each process, enforced by starting total/m processes.")
def movie(movie_type, path, **kwargs):
    """Movie of type MOVIE_TYPE from dumps at PATH.
    Generate the frames of a movie running over all dumps at the given path.
    """
    # Add these so we can pass on just the args
    kwargs['movie_type'] = movie_type
    kwargs['path'] = path
    # Try to load known filenames
    files = io.get_fnames(path)

    frame_dir = "frames_" + movie_type
    os.makedirs(frame_dir, exist_ok=True)

    try:
        # Load diagnostics from HARM itself
        diag = io.read_log(glob.glob(os.path.join(path, "*.hst")))
    except IOError:
        diag = None

    if 'debug' in kwargs and kwargs['debug']:
        # Run sequentially to make backtraces work
        for fname in files:
            frame(fname, diag, kwargs)
    else:
        # Try to guess how many processes before we MemoryError out
        if 'nthreads' not in kwargs or kwargs['nthreads'] is None:
            if 'memory_limit' in kwargs and kwargs['memory_limit'] is not None:
                hdr = io.read_hdr(files[0])
                nthreads = min(calc_nthreads(hdr, pad=0.6),
                            psutil.cpu_count())
            else:
                nthreads = psutil.cpu_count()
        # This application is entirely side-effects (frame creation)
        # So we map & ignore result
        # TODO pass single figure & blit/etc?
        if not use_mpi:
            print("Using {} processes".format(nthreads))
            with multiprocessing.Pool(nthreads) as pool:
                args = zip(files, (diag,)*len(files), (kwargs,)*len(files))
                pool.starmap_async(frame, args).get(720000)
        else:
            # Everything is set by MPI.  We inherit a situation and use it
            with MPICommExecutor() as executor:
                if executor is not None:
                    print("Imaging {} files in pool size {}".format(len(files), MPI.COMM_WORLD.Get_size()), file=sys.stderr)
                    executor.map(frame, files, (diag,)*len(files), (kwargs,)*len(files))

if __name__ == "__main__":
    movie()
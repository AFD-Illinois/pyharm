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

import pyHARM
from pyHARM.plots.frame import frame
from pyHARM.util import calc_nthreads

@click.command()
@click.argument('movie_type', nargs=1)
@click.argument('paths', nargs=-1, type=click.Path(exists=True))
# Common options
@click.option('-s', '--tstart', default=0, help="Start time.")
@click.option('-e', '--tend', default=1e7, help="End time.")
@click.option('-n', '--nthreads', default=None, help="Number of processes to use, if not using MPI")
# If you're invoking manually
@click.option('--fig_x', default=16, help="Figure width in inches.")
@click.option('--fig_y', default=9, help="Figure height in inches.")
@click.option('--fig_dpi', default=100, help="DPI of resulting figure.")
@click.option('-g','--plot_ghost', is_flag=True, default=False, help="Plot ghost zones.")
@click.option('-sz','--size', default=None, help="Window size, in M each side of central object.")
@click.option('-sh','--shading', default='gouraud', help="Shading: flat, nearest, gouraud.")
@click.option('--vmin', default=None, help="Colorscale minimum.")
@click.option('--vmax', default=None, help="Colorscale maximum.")
@click.option('--cmap', default='jet', help="Colormap.")
# Extras
@click.option('-r', '--resume', is_flag=True, default=False, help="Continue a previous run, by skipping existing frames")
@click.option('-d', '--debug', is_flag=True, default=False, help="Serial operation for debugging")
@click.option('-m', '--memory_limit', default=1, help="Memory limit in GB for each process, enforced by starting total/m processes.")
def movie(movie_type, paths, **kwargs):
    """Movie of type MOVIE_TYPE from dumps at each of PATHS.

    Each PATH can contain dump files in any format readable by pyHARM, either in the given directory or a subdirectory
    named "dumps", "dumps_kharma" or similar.

    "Movies" are generated as collections of frames in .png format, named frame_tXXXXXXXX.png by simulation time in M,
    and placed in a subdirectory "frames_MOVIE_TYPE" of the given PATH.  One can easily generate a single .mp4 movie
    from these using ffmpeg or similar.

    MOVIE_TYPE can be any variable known to pyHARM (see README, variables.py) or any "figure" function in figures.py.  A common
    first movie is 'log_rho' which will plot a phi=0 toroidal and midplane poloidal slice of the log_10 of the density. 

    If run within an MPI job/allocation with mpi4py installed, movie.py will attempt to use all allocated nodes to generate
    frames.  YMMV wildly with MPI installations, with mpi4py installed via pip generally a better choice than through conda.
    """
    base_path = os.getcwd()
    for path in paths:
        # change dir to path we want to image
        os.chdir(path)

        # Add these so we can pass on just the args
        kwargs['movie_type'] = movie_type
        kwargs['path'] = path
        # Try to load known filenames
        files = pyHARM.io.get_fnames(".")

        frame_dir = "frames_" + movie_type
        os.makedirs(frame_dir, exist_ok=True)

        try:
            # Load diagnostics from HARM itself
            fname = glob.glob(os.path.join(path, "*.hst"))[0]
            print("Loading diag file ",fname, file=sys.stderr)
            diag = pyHARM.io.read_log(fname)
        except IOError:
            diag = None

        if 'debug' in kwargs and kwargs['debug']:
            # Import profiling only if used, start it
            import cProfile, pstats, io
            from pstats import SortKey
            from pympler import tracker
            pr = cProfile.Profile()
            pr.enable()
            # Run sequentially to make profiling & backtraces work
            for fname in files:
                tr = tracker.SummaryTracker()
                frame(fname, diag, kwargs)
                tr.print_diff()
            
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.TIME)
            ps.print_stats(10)
            print(s.getvalue())

        else:
            # Try to guess how many processes before we MemoryError out
            if 'nthreads' not in kwargs or kwargs['nthreads'] is None:
                if 'memory_limit' in kwargs and kwargs['memory_limit'] is not None:
                    hdr = pyHARM.io.read_hdr(files[0])
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

        # Change back to wherever our root was, argument paths can be relative
        os.chdir(base_path)

if __name__ == "__main__":
    movie()

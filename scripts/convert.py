#!/usr/bin/env python3

import os
import psutil
import click
import multiprocessing

import pyHARM

def convert_file(fname, outfname):
    """Read a single file of some type, and write it as Illinois HDF5 format"""
    dump = pyHARM.load_dump(fname, grid_cache=False)
    # The zero is dt, which KHARMA does not keep
    pyHARM.io.iharm3d.write_dump(dump, outfname)

@click.command()
@click.argument('files', nargs=-1)
# Common options
@click.option('-p', '--path', default=None, help="Output filepath prefix, e.g. dumps/.")
@click.option('-o', '--outfile', default=None, help="Output filename, if converting a single dump.")
@click.option('-nt', '--nthreads', default=None, help="Number of parallel conversions -- defaults to nprocs")
@click.option('-d', '--debug', is_flag=True, help="Serial operation for debugging")
def convert(files, path, outfile, nthreads, debug):
    """Convert a file from KHARMA format (or any readable format) into iharm3d/Illinois HDF5 format
    (and only this format, as pyHARM cannot write other file formats).

    Tries to preserve as much as possible of the original data, but not everything may transfer:
    see pyHARM/io/iharm3d.py for details of the file writer.

    Defaults to writing output to the same directory as input files, optionally output to a different
    directory with '-p'. Adds a '.h5' file extension in place of KHARMA's .phdf, or additionally to other
    formats.

    Usage: convert.py FILE1 [FILE2 FILE3]
    """
    if path is not None:
        os.makedirs(path, exist_ok=True)
        outfiles = os.path.join(path, [f.split("/")[-1].replace(".phdf", "")+".h5" for f in files])
    elif outfile is not None:
        outfiles = [outfile]
    else:
        outfiles = [f.replace(".phdf", "")+".h5" for f in files]

    if debug:
        for file, outfile in zip(files, outfiles):
            convert_file(file, outfile)
    else:
        if nthreads is None:
            nthreads = min(psutil.cpu_count(), len(files))

        print("Converting {} files with {} threads".format(len(files), nthreads))
        with multiprocessing.Pool(nthreads) as pool:
            pool.starmap_async(convert_file, files, outfiles).get(720000)


if __name__ == "__main__":
    convert()
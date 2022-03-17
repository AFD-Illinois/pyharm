#!/usr/bin/env python3

# This script calculates analysis reductions over an entire run, i.e. every zone of every dump file.
# It makes use of multiprocessing, but the effect is limited due to high memory usage

# Usage: python analysis.py /path/to/dump/folder/ [tstart] tavg_start tavg_end [tend]
# All dump_*.h5 files in the path are assumed to be HARM fluid dump files
# tavg_* are the interval in time (r_g/c) over which to average any time-averaged variables
# EHT MAD comparison standard is 6000-10000M for a 10kM run. SANE comparison was 5-10kM
# * t* are the start and end times for all processing -- dumps before/after this will be ignored
# This is useful if multiple copies of analysis.py are being run and the results later merged
# (see merge_ana.py)

# Alternate usage: analysis.py /path/to/dump.h5
# Test analysis over just 1 dump file
# used to make sure it completes successfully/correctly before running at scale.

import os
import sys
from glob import glob
import numpy as np
import h5py

# Suppress runtime math warnings
# We will be dividing by zero, and don't want to hear about it
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# For memory usage stats
import psutil

import pyharm
# Spam base namespace so we don't have to type pyharm or even ph
from pyharm.variables import *
from pyharm.reductions import *

import pyharm.io as io
import pyharm.util as util

# Specific useful things 
from pyharm.defs import Loci
from pyharm.util import i_of

# Whether to augment or replace existing results
# Augmenting behaves *very* badly when jobs time out or are cancelled
# TODO add more file checks before appending to make augmenting more reliable
augment = False

# Whether to calculate each set of variables
# Once performed once, calculations will be ported to each new output file
calc_basic = True # Fluxes at horizon, often needed before movies, as basic check, etc.
calc_ravgs = False # Radial averages in the disk for code comparisons
calc_efluxes = False # Fluxes in places other than the horizon, to judge infall equilibrium etc.

# Stuff written specifically for the MAD code comparison
# A little long but useful
calc_madcc = False
calc_madcc_optional = False
# Field fluxes away from EH
calc_phi = False

# Maxima/minima over the grid that might prove useful diagnostics
calc_diagnostics = False

# Specialized calculations
calc_thavgs = False
calc_omega_bz = False
calc_jet_profile = False
calc_jet_cuts = False
calc_lumproxy = True
calc_gridtotals = False
calc_outfluxes = False

calc_pdfs = False
pdf_nbins = 200

params = {}
parallel = True

# This doesn't seem like the _right_ way to do optional args
# Skips everything before tstart, averages between tavg_start and tavg_end
tstart = None
tavg_start = None
tavg_end = None
tend = None
path = sys.argv[1]
if ".h5" not in path:
    dumps = io.get_fnames(path)
    if len(sys.argv) > 5:
        tstart = float(sys.argv[2])
        tavg_start = float(sys.argv[3])
        tavg_end = float(sys.argv[4])
        tend = float(sys.argv[5])
    elif len(sys.argv) > 3:
        tavg_start = float(sys.argv[2])
        tavg_end = float(sys.argv[3])
    else:
        print("Format: python analysis.py {dump_to_analyze.h5|/path/to/dumps/} [tstart] tavg_start tavg_end [tend]")
        sys.exit()
    debug = False
else:
    # Allow debugging new analysis over one dump with minimal arguments
    dumps = [path]
    path = os.path.dirname(path)
    tavg_start = 0
    tavg_end = 1e7
    debug = True




def avg_dump(n):
    out = {}

    if t < tstart or t > tend:
        # Still return the time
        return out

    print("Loading {} / {}: t = {}".format((n + 1), len(dumps), int(t)), file=sys.stderr)
    # TODO Add only what we need here...
    dump = pyharm.load_dump(dumps[n], params=params, calc_derived=True, add_jcon=False, add_fails=False, add_floors=False)

    # Should we compute the time-averaged quantities?
    do_tavgs = (tavg_start <= t <= tavg_end)

    # COMPUTATIONS

    this_process = psutil.Process(os.getpid())
    print("Memory use: {} GB".format(this_process.memory_info().rss / 10**9))

    del dump

    return out


def merge_dict(n, out, out_full):
    # Merge the output dicts, translate ending tags from above into HDF5 groups for easier merge/read
    for key in list(out.keys()):
        tag = key.split('/')[0]
        if key not in out_full:
            # Add the destination ndarray if not present
            # TODO this can probably be reduced to zeros(ND+out[key[-1]].shape)
            if tag == 'rt':
                out_full[key] = np.zeros((ND, hdr['n1']))
            elif tag == 'htht':
                out_full[key] = np.zeros((ND, hdr['n2'] // 2))
            elif tag == 'tht':
                out_full[key] = np.zeros((ND, hdr['n2']))
            elif tag == 'phit':
                out_full[key] = np.zeros((ND, hdr['n3']))
            elif tag == 'rtht':
                out_full[key] = np.zeros((ND, hdr['n1'], hdr['n2']))
            elif tag == 'thphit':
                out_full[key] = np.zeros((ND, hdr['n2'], hdr['n3']))
            elif tag == 'rphit':
                out_full[key] = np.zeros((ND, hdr['n1'], hdr['n3']))
            elif tag == 'pdft':
                out_full[key] = np.zeros((ND, pdf_nbins))
            elif tag in ['r', 'hth', 'rhth', 'th', 'phi', 'rth', 'rphi', 'thphi', 'pdf']:
                out_full[key] = np.zeros_like(out[key])
            else:
                out_full[key] = np.zeros(ND)
        # Average the averaged tags, slot in the time-dep tags
        if tag in ['r', 'hth', 'th', 'phi', 'rth', 'rhth', 'rphi', 'thphi', 'pdf']:
            # Weight the average correctly for _us_.  Full weighting will be done on merge w/the key 'avg/w'
            if my_avg_range > 0:
                try:
                    out_full[key][()] += out[key] / my_avg_range
                except TypeError as e:
                    print("Encountered error when updating {}: {}".format(key, e))
        else:
            try:
                if ND > 1:
                    out_full[key][n] = out[key]
                else:
                    # Array created above will only have 1D
                    out_full[key][:] = out[key]
            except TypeError as e:
                print("Encountered error when updating {}: {}".format(key, e))


# TODO this, properly, some other day
if ND < 200:
    nstart, nmin, nmax, nend = 0, 0, ND - 1, ND - 1
elif ND < 300:
    nstart, nmin, nmax, nend = 0, ND // 2, ND - 1, ND - 1
else:
    nstart, nmin, nmax, nend = int(tstart) // 5, int(tavg_start) // 5, int(tavg_end) // 5, int(tend) // 5

full_avg_range = nmax - nmin

if nmin < nstart: nmin = nstart
if nmin > nend: nmin = nend
if nmax < nstart: nmax = nstart
if nmax > nend: nmax = nend

my_avg_range = nmax - nmin

# If we're testing over just 1 dump, keep radial "averages" for reference
if full_avg_range < 1:
    full_avg_range = 1
    my_avg_range = 1

print("nstart = {}, nmin = {}, nmax = {} nend = {}".format(nstart, nmin, nmax, nend))

# Deduce the name of the output file
if tstart > 0 or tend < 10000:
    outfname = "eht_out_{:08d}_{:08d}.h5".format(int(tstart), int(tend))
else:
    outfname = "eht_out.h5"

# Open the output file -- note we keep it open the *whole time*, as we write each dump's
# results ~immediately to save memory. (TODO could we write per-process with a lock?)
# Delete the current one unless we're specially "augmenting" current results to save time
if not augment:
    outf = h5py.File(outfname, 'w')
    print("Replacing existing output: {}".format(outfname))
else:
    try:
        # This will append if file exists, otherwise create
        outf = h5py.File(outfname, 'a')
    except OSError:
        # If it exists but isn't valid HDF5, there's no saving it
        # Truncate it instead
        outf = h5py.File(outfname, 'w')
        print("Replacing existing output: {}".format(outfname))

inf = h5py.File(dumps[0],'r')
if 'header' in inf:
    hdr_preserve = io.hdr.hdf5_to_dict(inf['header'])
    if not 'header' in outf:
        outf.create_group('header')
    io.hdr.dict_to_hdf5(hdr_preserve, outf['header'])
inf.close()

# Fill the output dict with all per-dump or averaged stuff
# Hopefully in a way that doesn't keep too much of it around in memory
if parallel:
    #nthreads = util.calc_nthreads(hdr, n_mkl=16, pad=0.23)
    nthreads = 5
    util.iter_parallel(avg_dump, merge_dict, outf, ND, nthreads)
else:
    for n in range(ND):
        out = avg_dump(n)
        merge_dict(n, out, outf)

# Toss in anything else we want to keep, including all the diagnostics
vars = {'avg/start': tavg_start,
        'avg/end': tavg_end,
        'avg/w': my_avg_range / full_avg_range}
diag = io.load_log(path)
# Move diags into a subfolder
if diag is not None:
    for key in diag:
        vars['diag/'+key] = diag[key]

for key in vars:
    if key not in outf:
        outf[key] = vars[key]
    else:
        try:
            outf[key][()] = vars[key]
        except TypeError as e:
            print("Error adding diag {} from HARM log to outfile: {}".format(key, e))

print("Merge operation will weight averages by {}".format(outf["avg/w"][()]))

outf.close()

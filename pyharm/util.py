################################################################################
#                                                                              #
#  UTILITY FUNCTIONS                                                           #
#                                                                              #
################################################################################

import glob
import os
import multiprocessing
import numpy as np
try:
    import psutil
    using_psutil = True
except ModuleNotFoundError as e:
    print("Not using psutil: ", e)
    using_psutil = False

# Run a function in parallel with Python's multiprocessing
# 'function' must take only a number
def run_parallel(function, nmax, nthreads, debug=False):
    # TODO if debug...
    pool = multiprocessing.Pool(nthreads)
    try:
        pool.map_async(function, list(range(nmax))).get(720000)
    except KeyboardInterrupt:
        print('Caught interrupt!')
        pool.terminate()
        exit(1)
    else:
        pool.close()
    pool.join()

# Run a function in parallel with Python's multiprocessing
# 'function' must take only a number
def map_parallel(function, nmax, nthreads, debug=False, initializer=None, initargs=()):
    if initializer is not None:
        pool = multiprocessing.Pool(nthreads, initializer=initializer, initargs=initargs)
    else:
        pool = multiprocessing.Pool(nthreads)

    try:
        # Map the function over the list. Results are 
        out_iter = pool.map(function, list(range(nmax)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    else:
        pool.close()
        pool.join()
    return out_iter

# Run a function in parallel with Python's multiprocessing
# 'function' must take only a number
# 'merge_function' must take the same number plus whatever 'function' outputs, and adds to the dictionary out_dict
def iter_parallel(function, merge_function, out_dict, nmax, nthreads, debug=False, initializer=None, initargs=()):
    if initializer is not None:
        pool = multiprocessing.Pool(nthreads, initializer=initializer, initargs=initargs)
    else:
        pool = multiprocessing.Pool(nthreads)

    try:
        # Map the above function to the dump numbers, returning an iterator of 'out' dicts to be merged one at a time
        # This avoids keeping the (very large) full pre-average list in memory
        out_iter = pool.imap(function, list(range(nmax)))
        for n, result in enumerate(out_iter):
            merge_function(n, result, out_dict)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    else:
        pool.close()
        pool.join()


# Calculate ideal # threads
# Lower pad values are safer
def calc_nthreads(hdr, n_mkl=8, pad=0.25):
    # Limit threads for 192^3+ problem due to memory
    if using_psutil:
        # Roughly compute memory and leave some generous padding for multiple copies and Python games
        # (N1*N2*N3*8)*(NPRIM + 4*4 + 6) = size of "dump," (N1*N2*N3*8)*(2*4*4 + 6) = size of "geom"
        # TODO get a better model for this!!
        ncopies = hdr['n_prim'] + 4 * 4 + 6
        nproc = int(pad * psutil.virtual_memory().total / (hdr['n1'] * hdr['n2'] * hdr['n3'] * 8 * ncopies))
        if nproc < 1:
            nproc = 1
    else:
        print("psutil not available: Using 4 processes as a safe default")

    return nproc

def slice_to_index(current_start, current_stop, slc):
    """Take a slice out of a range represented by start and end points.
    Resolves positive, negative, and integer slice values correctly.
    """
    new_start = list(current_start).copy()
    new_stop = list(current_stop).copy()
    for i in range(len(slc)):
        if isinstance(slc[i], int):
            new_start[i] = current_start[i] + slc[i]
            new_stop[i] = current_start[i] + slc[i] + 1
        elif slc[i] is not None:
            if slc[i].start is not None:
                new_start[i] = current_start + slc[i].start if slc[i].start > 0 else current_stop[i] + slc[i].stop
            if slc[i].stop is not None:
                # Count forward from global_start, or backward from global_stop
                new_stop[i] = current_start + slc[i].stop if slc[i].stop > 0 else current_stop[i] + slc[i].stop

    return new_start, new_stop

# Convenience for finding zone containing a given value,
# in coordinate/monotonic-increase variables
def i_of(var, val, behind=True, fail=False):
    """Convenience for finding zone containing a given value,
    in coordinate/monotonic-increase variables
    """
    i = 0
    while var[i] < val:
        i += 1
        # Warn or fail if we step too far
        if i == len(var):
            if fail:
                raise ValueError("Array does not contain value {}".format(val))
            else:
                print("Warning: using last value {} as desired value {}".format(var[-1], val))
                break

    # Return zone before the value, usually what we want for fluxes
    if behind or i == len(var):
        i -= 1

    return i

__license__ = """
 File: parallel.py
 
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

import multiprocessing
from tqdm.auto import tqdm
try:
    import psutil
    using_psutil = True
except ModuleNotFoundError as e:
    print("Not using psutil: ", e)
    using_psutil = False

# Hack for passing lambdas to a parallel function:
# Initialize with a dangling function pointer and fill it at call time
# Credit https://medium.com/@yasufumy/python-multiprocessing-c6d54107dd55
_func = None

__doc__ = \
"""Tools for running embarrassingly parallel operations using multiple processes.
"""

def _worker_init(func):
    global _func
    _func = func

def _worker(x):
    return _func(x)

def map_parallel(function, input_list, nprocs=None):
    """Run a function in parallel and return a list of all the results. Best for whole-image reductions.
    Takes lambdas thanks to some happy hacking
    """
    input_list = list(input_list)
    with multiprocessing.Pool(nprocs, initializer=_worker_init, initargs=(function,)) as p:
        return list(tqdm(p.imap(_worker, input_list), total=len(input_list)))


def iter_parallel(function, merge_function, input_list, output, nprocs=None, initializer=None, initargs=()):
    """Run a function in parallel with Python's multiprocessing, applied either independently or as a reduction,
    depending on the implementation of ``merge_function``.

    :param function: function to run. Must not be a lambda, must take a single element of ``input_list``.
    :param merge_function: function merging outputs. Must take the list element number, the output of ``function``,
                            and whatever is passed as ``output``, to be used as an accumulator (or list of results).
                            Also cannot be a lambda as it has a side effect, but can be defined locally.
    :param input_list: list of input filenames/images/whatever
    :param output: variable or list of appropriate size for writing results
    """
    if initializer is not None:
        pool = multiprocessing.Pool(nprocs, initializer=initializer, initargs=initargs)
    else:
        pool = multiprocessing.Pool(nprocs)

    try:
        # Map the above function to the dump numbers, returning an iterator of 'out' dicts to be merged one at a time
        # This avoids keeping the (very large) full pre-average list in memory
        out_iter = pool.imap(function, input_list)
        for n, result in enumerate(out_iter):
            merge_function(n, result, output)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    else:
        pool.close()
        pool.join()

def calc_nthreads(hdr, n_mkl=8, pad=0.25):
    """Calculate a reasonable number of threads for the problem size based on available RAM.
    Necessarily varying degrees of unreliable in predicting true usage -- hence padding parameter for safety.
    Note pad is a proportion, so LOWER pad values are safer.
    """
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

def set_mkl_threads(n_mkl):
    """Try to set the MKL numpy backend to use ``n_mkl`` threads."""
    try:
        import ctypes
        mkl_rt = ctypes.CDLL('libmkl_rt.so')
        mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
        mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads
        mkl_set_num_threads(n_mkl)
        print("Using {} MKL threads".format(mkl_get_max_threads()))
    except Exception as e:
        print(e)
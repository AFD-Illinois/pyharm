__license__ = """
 File: reductions.py
 
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

import numpy as np
import scipy.fftpack as fft

# This is too darn useful
from pyharm.util import i_of

__doc__ = \
"""The general interface for reductions is (dump, var, options)
| dump: FluidState object, generally pre-slice.
| var: either the name of a variable to compute over the necessary slice,
or a pre-sliced & computed variable of the right shape.

Other options as described below.
Certain zone arguments with defaults are still necessary when a reduction is otherwise ambiguous!
"""

# TODO PDFs. Any stray incompatibilities with new slicing stuff
# TODO would need lots of modification for non-regular or not-in-native coordinates
# TODO slice-before-calculate support for correlation functions

## Plotting reductions ##

def flatten_xz(dump, var, at=None, sum=False, half_cut=False):
    """Return an X-Z slice or sum of var, generally for use in making a plot.
    By default takes both the 0-degree (right side) and 180-degree (left side) slices,
    to make a full slice across the pole.
    Note sums are *not* GR-aware!

    :param at: which rank in X3 to take data from
    :param sum: whether to sum all ranks. Overrides 'at'
    :param half_cut: just take the rank 'at', not its 180-degree opposite
    """
    if sum:
        if isinstance(var, str):
            var = dump[var]
        if len(var.shape) == 3:
            var = var.sum(-1)
        if half_cut:
            return var
        else:
            return np.append(var, np.flip(var, 1), 1)
    else:
        if at is None:
            at = 0
        if isinstance(var, str):
            if half_cut:
                return np.squeeze(dump[:, :, at][var])
            else:
                return np.append(np.squeeze(dump[:, :, at][var]), np.flip(np.squeeze(dump[:, :, at + dump['n3']//2][var]), 1), 1)
        else:
            if half_cut:
                if len(var.shape) == 3:
                    return var[:, :, at]
                else:
                    return var
            else:
                return np.append(var[:, :, at], np.flip(var[:, :, at + dump['n3']//2], 1), 1)

def flatten_xy(dump, var, at=None, sum=False):
    """Return an X-Y slice or sum of var.  Note sums are *not* GR-aware!

    :param at: which rank in X2 to take data from, default N2//2
    :param sum: whether to sum all ranks. Overrides 'at'.
    """
    if sum:
        if isinstance(var, str):
            var = dump[var]
        return var.sum(1)
    else:
        if at is None:
            at = dump['n2']//2
        if isinstance(var, str):
            return np.squeeze(dump[:, at, :][var])
        else:
            return var[:, at, :]

def wrap(x):
    """Append the first rank in axis 1 again after the last rank in axis 1
    (i.e. the angular index, after any reductions).

    Useful when plotting zone-centered th or phi variables, as it allows shading
    of the region between ends of the array eliminating an ugly gap.
    """
    return np.append(x[Ellipsis], x[:, 0:1], 1)

def flatten_thphi(dump, var, at=5, sum=False):
    """Return a th-phi slice or sum of var.  Note sums are *not* GR-aware!

    :param at: which rank in X1 to take data from.
    :param sum: whether to sum all ranks. Overrides 'at'.
    """
    if sum:
        if isinstance(var, str):
            var = dump[var]
        return var.sum(0)
    else:
        if isinstance(var, str):
            return np.squeeze(dump[at][var])
        else:
            return var[at]

## Slices for below ##
# Functions to get indices, given locations in r,th,phi
# TODO does this belong in Grid?  General i_of/j_of/k_of?
def get_i_slice(dump, r_min, r_max):
    """Calculate slice imin:imax for a given range in r"""
    return slice(i_of(dump['r1d'], r_min), i_of(dump['r1d'], r_max))

def get_j_bounds(dump, th_min=np.pi / 3., th_max=2*np.pi / 3.):
    """Calculate slice jmin:jmax for a given range in theta: returns *tuple*
    Defaults to EHT disk profile: pi/3 to 2pi/3
    """
    return (i_of(dump['th1d'], th_min), i_of(dump['th1d'], th_max))

def get_j_slice(dump, th_min=np.pi / 3., th_max=2*np.pi / 3.):
    """Calculate slice jmin:jmax for a given range in theta: returns *slice*
    Defaults to EHT disk profile: pi/3 to 2pi/3
    """
    return slice(*get_j_bounds(dump, th_min, th_max))

## Sums and Averages ##
# Generally GR-aware sum, average, or combination operations producing 1D profiles in r, th.

def shell_sum(dump, var, at_r=None, at_i=None, th_slice=None, j_slice=None, mask=None):
    """Sum a variable over spherical shells. Returns a radial profile (array length N1) or single-shell sum

    :param at_r: Single radius at which to sum (nearest-neighbor smaller zone is used)
    :param at_zone: Specific radial zone at which to sum, for compatibility
    :param th_slice: Tuple of minimum and maximum theta value to sum
    :param j_slice: Tuple of x2 indices instead of specifying theta. Overrides th_slice
    :param mask: array of 1/0 of the post-slice size of 'var', which is multiplied with the result
    """

    # Translate coordinates to zone numbers.
    if at_i is not None:
        i_slice = slice(at_i, at_i+1)
    elif at_r is not None:
        at_i = i_of(dump['r1d'], at_r)
        i_slice = slice(at_i, at_i+1)
    else:
        i_slice = slice(None)

    if j_slice is not None:
        j_slice = slice(j_slice[0], j_slice[1])
    elif th_slice is not None:
        j_slice = get_j_slice(dump, th_slice[0], th_slice[1])
    else:
        j_slice = slice(None)

    if isinstance(var, str):
        var = dump[i_slice, j_slice, :][var]
    else:
        var = var[i_slice, j_slice, :]

    integrand = var * dump['gdet'][i_slice, j_slice, :] * dump['dx2'] * dump['dx3']
    if mask is not None:
        integrand *= mask

    # This should usually return the right thing:
    # 1d array in r, or 0d array (~= scalar) for single shell 
    return np.squeeze(np.sum(integrand, axis=(1, 2)))


def shell_avg(dump, var, **kwargs):
    """Average a variable over spherical shells. Returns a radial profile (array length N1) or single-shell average.
    See shell_sum for arguments.
    """
    return shell_sum(dump, var, **kwargs) / shell_sum(dump, '1', **kwargs)


def sphere_sum(dump, var, r_slice=None, i_slice=None, th_slice=None, j_slice=None, mask=None):
    """Sum everything within a sphere, semi-sphere, or thick spherical shell.
    Extents are specified optionally in r or i, and th or j, with indices taking precedence
    Mask is multiplied at the end
    """
    # Translate coordinates to zone numbers.
    if i_slice is not None:
        i_slice = slice(i_slice[0], i_slice[1])
    elif r_slice is not None:
        i_slice = get_i_slice(r_slice[0], r_slice[1])
    else:
        i_slice = slice(None)

    if j_slice is not None:
        j_slice = slice(j_slice[0], j_slice[1])
    elif th_slice is not None:
        j_slice = get_j_slice(dump, th_slice[0], th_slice[1])
    else:
        j_slice = slice(None)

    if isinstance(var, str):
        var = dump[i_slice, j_slice, :][var]
    else:
        var = var[i_slice, j_slice, :]

    # TODO mask support?
    return np.sum(var * dump['gdet'][i_slice, j_slice, :] * dump['dx1'] * dump['dx2'] * dump['dx3'])


def sphere_avg(dump, var, **kwargs):
    """Average everything within a sphere, semi-sphere or thick spherical shell.
    See sphere_sum for arguments.
    """
    return sphere_sum(dump, var, **kwargs) / sphere_sum(dump, '1', **kwargs)


def midplane_sum(dump, var, zones=2, **kwargs):
    """Average a few zones adjacent to midplane, then sum.
    Allows specifying an r_slice or i_slice within which to sum.
    """
    jmin = dump['n2'] // 2 - zones//2
    jmax = dump['n2'] // 2 + zones//2
    return sphere_sum(dump, var, j_slice=(jmin, jmax), **kwargs) / (jmax - jmin)


def theta_profile(dump, var, start, zones_to_av=1, use_gdet=True, fold=True):
    """Profile in theta by averaging over phi at a particular radius (or average among a few close radii).
    Note that this function returns an array of size N2 if fold==False, N2//2 if fold==True
    
    :param start: zone number to average at
    :param zones_to_av: number of zones to average, starting at 'start'
    :param use_gdet: whether or not to make averaging GR-aware
    :param fold: whether to consider the system as symmetric about the midplane and average hemispheres
    """
    i_slice = (slice(start, start+zones_to_av), slice(None), slice(None))

    # Slices representing hemispheres starting from the poles
    j_top = slice(None, dump['n2']//2)
    j_bottom = slice(None, dump['n2']//2-1, -1)

    if use_gdet:
        jacobian = dump['gdet'][i_slice] * dump['dx1'] * dump['dx3']
        if isinstance(var, str):
            integrand = dump[i_slice][var] * jacobian
        else:
            integrand = var[i_slice] * jacobian

        if fold:
            return (integrand[:,j_top].sum(axis=(0, 2)) + integrand[:,j_bottom].sum(axis=(0, 2))) / (2*jacobian[:,j_top].sum(axis=(0,2)))
        else:
            return integrand.sum(axis=(0, 2)) / jacobian.sum(axis=(0,2))
    else:
        if isinstance(var, str):
            integrand = dump[i_slice][var]
        else:
            integrand = var[i_slice]

        if fold:
            return (integrand[:,j_top].mean(axis=(0, 2)) + integrand[:,j_bottom].mean(axis=(0, 2))) / 2
        else:
            return integrand.mean(axis=(0, 2))


## Correlation functions/lengths ##

def corr_midplane(var, norm=True, at_i1=None):
    """Angular correlation function at the midplane,
    of an array representing a variable in spherical-like coordinates r,th,phi
    """
    if at_i1 is None:
        at_i1 = range(var.shape[0])
    if isinstance(at_i1,int):
        at_i1 = (at_i1,)

    # This selects the midplane-adjacent zones N2/2-1 & N2/2
    jmin = var.shape[1] // 2 - 1
    jmax = var.shape[1] // 2 + 1

    R = np.zeros((len(at_i1), var.shape[2]))

    # TODO is there a way to vectorize over R? Also, are we going to average over adjacent r ever?
    for i_out,i1 in enumerate(at_i1):
        # Average over small angle around midplane
        var_phi = np.mean(var[i1, jmin:jmax, :], axis=0)
        # Calculate autocorrelation
        var_phi_normal = (var_phi - np.mean(var_phi)) / np.std(var_phi)
        var_corr = fft.ifft(np.abs(fft.fft(var_phi_normal)) ** 2)
        R[i_out] = np.real(var_corr) / var_corr.size

    if norm:
        normR = R[:, 0]
        for k in range(var.shape[2]):
            R[:, k] /= normR

    if R.shape[0] == 1:
        return R[0,:]
    else:
        return R


def corr_midplane_direct(var, norm=True):
    """Alternate more volatile implementation of corr_midplane
    """
    jmin = var.shape[1] // 2 - 1
    jmax = var.shape[1] // 2 + 1

    var_norm = np.ones((var.shape[0], 2, var.shape[2]))
    # Normalize radii separately
    for i in range(var.shape[0]):
        vmean = np.mean(var[i, jmin:jmax, :])
        var_norm[i, :, :] = var[i, jmin:jmax, :] - vmean

    R = np.ones((var.shape[0], var.shape[2]))
    for k in range(var.shape[2]):
        R[:, k] = np.sum(var_norm * np.roll(var_norm, k, axis=-1), axis=(1, 2)) / 2

    if norm:
        normR = R[:, 0]
        for k in range(var.shape[2]):
            R[:, k] /= normR

    return R


def corr_length_phi(R, interpolate=True):
    """Correlation "length" (angle) given a correlation function of r,phi"""
    lam = np.zeros(R.shape[0])
    for i in range(R.shape[0]):
        k = 0
        target = R[i, 0] / np.exp(1)
        while k < R.shape[1] and R[i, k] >= target:
            k += 1

        if interpolate and k < R.shape[1]:
            k = (target + (k - 1) * R[i, k] - k * R[i, k - 1]) / (R[i, k] - R[i, k-1])

        # This is specific to phi
        lam[i] = k * (2 * np.pi / R.shape[1])
    return lam


## Power Spectra ##

def pspec(var, dt=5, window=0.33, half_overlap=False, bin="fib"):
    """Power spectrum of a 1D timeseries, with various windows/binning.
    Currently only supports equally spaced times dt.
    """
    if not np.any(var[var.size // 2:]):
        return np.zeros_like(var), np.zeros_like(var)

    data = var[var.size // 2:]
    data = data[np.nonzero(data)] - np.mean(data[np.nonzero(data)])

    if window < 1:
        window = int(window * data.size)
    print("FFT window is {} of {} samples".format(window, data.size))

    print("Sampling time is {}".format(dt))
    out_freq = np.abs(fft.fftfreq(window, dt))

    if half_overlap:
        # Hanning w/50% overlap
        spacing = (window // 2)
        nsamples = data.size // spacing

        out = np.zeros(window)
        for i in range(nsamples - 1):
            windowed = np.hanning(window) * data[i * spacing:(i + window // spacing) * spacing]
            out += np.abs(fft.fft(windowed)) ** 2

        # TODO binning?

        freqs = out_freq

    else:
        # Hamming no overlap, like comparison paper
        nsamples = data.size // window

        for i in range(nsamples):
            windowed = np.hamming(window) * data[i * window:(i + 1) * window]
            pspec = np.abs(fft.fft(windowed)) ** 2

            # Bin data, declare accumulator output when we know its size
            if bin == "fib":
                # Modify pspec, allocate for modified form
                pspec, freqs = _fib_bin(pspec, out_freq)

                if i == 0:
                    out = np.zeros_like(np.array(pspec))
            else:
                if i == 0:
                    out = np.zeros(window)

            out += pspec

    print("PSD using ", nsamples, " segments.")
    out /= nsamples
    out_freq = freqs

    return out_freq, out


def _fib_bin(data, freqs):
    """Fibonacci sequence for binning.
    It is somehow mildly concerning that this is a thing that works.
    """
    j = 0
    fib_a = 1
    fib_b = 1
    pspec = []
    pspec_freq = []
    while j + fib_b < data.size:
        pspec.append(np.mean(data[j:j + fib_b]))
        pspec_freq.append(np.mean(freqs[j:j + fib_b]))
        j = j + fib_b
        fib_c = fib_a + fib_b
        fib_a = fib_b
        fib_b = fib_c

    return np.array(pspec), np.array(pspec_freq)

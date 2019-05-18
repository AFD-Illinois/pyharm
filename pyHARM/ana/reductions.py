# Data reductions (integrals, averages, statistics) for fluid simulation output data

import numpy as np
# TODO pystella version?
import scipy.fftpack as fft


# General interface for reductions:
# (dump, var, options)
# dump: IharmDump object (or anything which provides dump['gdet'] and dump.header['nx'] members).  Sometimes omitted.
# var: ndarray of 3 dimensions, or a vector with vector index //first// (mu,i,j,k)
# other options as described below.  Certain zone arguments are necessary when a reduction is otherwise ambiguous

# TODO PDFs


## Correlation functions/lengths ##


def corr_midplane(var, norm=True, at_i1=None):
    """Angular correlation function at the midplane,
     of an array representing a variable in spherical-like coordinates r,th,phi
     """
    if at_i1 is None:
        at_i1 = range(var.shape[0])

    jmin = var.shape[1] // 2 - 1
    jmax = var.shape[1] // 2 + 1

    R = np.zeros((len(at_i1), var.shape[2]))

    # TODO is there a way to vectorize over R? Also, are we going to average over adjacent r ever?
    for i1 in at_i1:
        # Average over small angle around midplane
        var_phi = np.mean(var[i1, jmin:jmax, :], axis=0)
        # Calculate autocorrelation
        var_phi_normal = (var_phi - np.mean(var_phi)) / np.std(var_phi)
        var_corr = fft.ifft(np.abs(fft.fft(var_phi_normal)) ** 2)
        R[i1] = np.real(var_corr) / var_corr.size

    if norm:
        normR = R[:, 0]
        for k in range(var.shape[2]):
            R[:, k] /= normR

    return R


def corr_midplane_direct(var, norm=True):
    """Alternate more volatile implementation of corr_midplane"""
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

    return out, out_freq


def _fib_bin(data, freqs):
    # Fibonacci binning.  Why is this a thing.
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


## Sums and Averages ##


def get_eht_disk_j_vals(dump):
    """Calculate jmin, jmax in theta coordinate for EHT disk profiles (pi/3 - 2pi/3)"""
    THMIN = np.pi / 3.
    THMAX = 2. * np.pi / 3.
    # Calculate jmin, jmax for EHT radial profiles
    #ths = G.coords.th(G.coord_all())[-1, :, 0]
    ths = dump['th'][-1, :, 0]
    jmin, jmax = -1, -1
    for n in range(len(ths)):
        if ths[n] > THMIN:
            jmin = n
            break

    for n in range(len(ths)):
        if ths[n] > THMAX:
            jmax = n
            break

    return jmin, jmax


def shell_sum(dump, var, at_zone=None, mask=None):
    """Sum a variable over a spherical shell at a particular X1 zone, or at each zone as a radial profile"""
    if isinstance(var, str):
        var = dump[var]

    # TODO could maybe be made faster with 'where' but also harder to get right
    integrand = var * dump['gdet'][:, :, None] * dump.header['dx2'] * dump.header['dx3']
    if mask is not None:
        integrand *= mask

    if at_zone is not None:
        return np.sum(integrand[at_zone, :, :], axis=(0, 1))
    else:
        return np.sum(integrand, axis=(1, 2))


def partial_shell_sum(dump, var, jmin, jmax):
    """Version of above sum limited to area between jmin/jmax, usually used for isolating disk/jet"""
    if isinstance(var, str):
        var = dump[var]

    return (var[:, jmin:jmax, :] *
            dump['gdet'][:, jmin:jmax, None] * dump.header['dx2'] * dump.header['dx3']).sum(axis=(1, 2)) / \
           ((dump['gdet'][:, jmin:jmax] * dump.header['dx2']).sum(axis=1) * 2 * np.pi)


def midplane_sum(dump, var, within=None):
    """Average the two zones adjacent to midplane and sum, optionally within some zone in X1"""
    if isinstance(var, str):
        var = dump[var]

    jmin = var.shape[1] // 2 - 1
    jmax = var.shape[1] // 2 + 1
    if within is not None:
        return np.sum(var[:within, jmin:jmax, :] *
                      dump['gdet'][:within, jmin:jmax, None] * dump.header['dx1'] * dump.header['dx3']) / \
               (jmax - jmin)
    else:
        return np.sum(var[:, jmin:jmax, :] *
                      dump['gdet'][:, jmin:jmax, None] * dump.header['dx1'] * dump.header['dx3']) / \
               (jmax - jmin)


def full_vol_sum(dump, var, within=None):
    """Sum variable over all space, or within the indicated zone in X1"""
    if isinstance(var, str):
        var = dump[var]

    if within is not None:
        return np.sum(var[:within, :, :] *
                      dump['gdet'][:within, :, None] * dump.header['dx1'] * dump.header['dx2'] * dump.header['dx3'])
    else:
        return np.sum(var * dump['gdet'][:, :, None] * dump.header['dx1'] * dump.header['dx2'] * dump.header['dx3'])


def partial_vol_sum(dump, var, jmin, jmax, outside=None):
    if isinstance(var, str):
        var = dump[var]

    # TODO can I cache the volume instead of passing jmin, jmax?
    if outside is not None:
        return np.sum(var[outside:, jmin:jmax, :] * dump['gdet'][outside:, jmin:jmax, None] *
                      dump.header['dx1'] * dump.header['dx2'] * dump.header['dx3'])
    else:
        return np.sum(var[:, jmin:jmax, :] * dump['gdet'][:, jmin:jmax, None] *
                      dump.header['dx1'] * dump.header['dx2'] * dump.header['dx3'])


def theta_av(dump, var, start, zones_to_av=1, use_gdet=False, fold=True):
    """Profile in theta by averaging over phi, and optionally also:
    hemispheres: set fold=True
    radial zones: set zones_to_av > 1
    """
    if isinstance(var, str):
        var = dump[var]

    N2 = var.shape[1]
    if use_gdet:
        # TODO currently implies fold -- also does this do anything different from below?
        return (var[start:start + zones_to_av, :N2//2, :] * dump['gdet'][start:start + zones_to_av, :N2//2, None] *
                dump.header['dx1'] * dump.header['dx3'] +
                var[start:start + zones_to_av, :N2//2-1:-1, :] *
                dump['gdet'][start:start + zones_to_av, :N2//2-1:-1, None] * dump.header['dx1'] * dump.header['dx3']
                ).sum(axis=(0, 2)) \
               / ((dump['gdet'][start:start + zones_to_av, :N2 // 2] * dump.header['dx1']).sum(axis=0) * 2 * np.pi)
    else:
        if fold:
            return (var[start:start + zones_to_av, :N2 // 2, :].mean(axis=(0, 2)) +
                    var[start:start + zones_to_av, :N2 // 2 - 1:-1, :].mean(axis=(0, 2))) / 2
        else:
            return var[start:start + zones_to_av, :, :].mean(axis=(0, 2))

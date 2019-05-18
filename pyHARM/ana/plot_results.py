# Functions for plotting analysis results

import numpy as np
import matplotlib

# TODO catalogue y-axis names for reduction/var combos?  Or too much work?
#pretty_names = {'': }


def plot_t(ax, ivar, var, range=(5000, 10000), label=None, xtick_labels=False):
    """Plot a variable vs time.  Separated because xticks default to false"""
    slc = np.nonzero(var)

    if label is not None:
        ax.plot(ivar[slc], var[slc], label=label)
    else:
        ax.plot(ivar[slc], var[slc])

    ax.set_xlim(range)
    if not xtick_labels:
        ax.set_xticklabels([])


def fit(x, y, log=False):
    if log:
        coeffs = np.polyfit(np.log(x), np.log(y), deg=1)
    else:
        coeffs = np.polyfit(x, y, deg=1)

    poly = np.poly1d(coeffs)
    if log:
        yfit = lambda xf: np.exp(poly(np.log(xf)))
    else:
        yfit = poly

    if log:
        fit_lab = r"{:.2g} * r^{:.2g}".format(np.exp(coeffs[1]), coeffs[0])
    else:
        fit_lab = r"{:2g}*x + {:2g}".format(coeffs[0], coeffs[1])

    return x, yfit(x), fit_lab
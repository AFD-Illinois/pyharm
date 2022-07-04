# Functions for plotting analysis results

import numpy as np
import matplotlib

from ..ana_results import AnaResults
from .pretty import pretty

def plot_hst(ax, diag, var, tline=None, xticklabels=None, xlabel=None, **kwargs):
    """Plot a scalar vs t, optionally marking with a red line representing current time"""

    ax.plot(*diag.get_result('t', var), label=pretty(var), **kwargs)

    if tline is not None:
        ax.axvline(tline, color='r')

    ax.legend(loc='upper left')
    ax.grid(True)

    ax.set_xlim((diag['t'][0], diag['t'][-1]))

    # This will be the easier way to add whatever
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

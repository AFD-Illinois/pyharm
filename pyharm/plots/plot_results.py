# Functions for plotting analysis results

import numpy as np
import matplotlib

from ..ana_results import AnaResults
from .pretty import pretty

diag_fn_dict = {
    'mdot': lambda diag: np.abs(diag['Mdot_EH_Flux']),
    'Mdot': lambda diag: np.abs(diag['Mdot_EH_Flux']),
    'phi_b': lambda diag: diag['Phi_EH']/np.sqrt(np.abs(diag['Mdot_EH_Flux']))
}

def plot_hst(ax, diag, var, tline=None, xticklabels=None, xlabel=None, **kwargs):
    if isinstance(var, str):
        vname = var
        if var in diag_fn_dict:
            var = diag_fn_dict[var](diag)
        else:
            var = diag[var]
    else:
        vname = ""
    ax.plot(diag['time'], var, label=pretty(vname), **kwargs)
    ax.axvline(tline, color='r')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_xlim((diag['time'][0], diag['time'][-1]))

    # This will be the easier way to add whatever
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

# Functions for reading old or non-HDF5 files for analysis

import os
import numpy as np


def load_log(path):
    # TODO specify log name in dumps, like grid
    logfname = os.path.join(path, "log.out")
    if not os.path.exists(logfname):
        return None
    dfile = np.loadtxt(logfname).transpose()

    # TODO log should probably have a header
    diag = {}
    diag['t'] = dfile[0]
    diag['rmed'] = dfile[1]
    diag['pp'] = dfile[2]
    diag['e'] = dfile[3]
    diag['uu_rho_gam_cent'] = dfile[4]
    diag['uu_cent'] = dfile[5]
    diag['mdot'] = dfile[6]
    diag['edot'] = dfile[7]
    diag['ldot'] = dfile[8]
    diag['mass'] = dfile[9]
    diag['egas'] = dfile[10]
    diag['Phi'] = dfile[11]
    diag['phi'] = dfile[12]
    diag['jet_EM_flux'] = dfile[13]
    diag['divbmax'] = dfile[14]
    diag['lum_eht'] = dfile[15]
    diag['mdot_eh'] = dfile[16]
    diag['edot_eh'] = dfile[17]
    diag['ldot_eh'] = dfile[18]

    return diag


# For adding contents of the log to dumps
def log_time(diag, var, t):
    if len(diag['t'].shape) < 1:
        return diag[var]
    else:
        i = 0
        while i < len(diag['t']) and diag['t'][i] < t:
            i += 1
        return diag[var][i - 1]

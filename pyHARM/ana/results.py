# Functions indexing a results directory

import os
import numpy as np

from pyHARM.io import hdf5_to_dict, read_hdr
from pyHARM.grid import Grid
from pyHARM.util import i_of
from pyHARM.ana.variables import fns_dict
import pyHARM.parameters as parameters
"""
Tools for dealing with the results computed by scripts/analysis.py.  Results are organized by remaining independent
variable -- so, a phi- and time-average will be under 'rth' since these are its remaining independent variables.

This is more logical than it sounds: this way, most quantity names contain all the info needed to plot them.
Also, multiple reductions of the same quantity can be stored with logical names, e.g. a phi,t average of rho in
'rth/rho' and a further average over th in 'r/rho'.

The one exception is variables containing th but not r.  These must have a radius specified, at least in FMKS coords
where th is mildly r-dependent.
We ignore this for informal plots, but for figures it can be specified with 'at_r'
"""

diag_fns = {'mdot': lambda diag: diag['t/Mdot'][()],
            'phi_b': lambda diag: diag['t/Phi_b'][()] / np.sqrt(diag['t/Mdot'][()]),
                            #np.mean(np.sqrt(diag['Mdot'][len(diag['Mdot'])//2:])),
            'edot': lambda diag: diag['t/Edot'][()] / diag['t/Mdot'][()],
                            #np.mean(diag['Mdot'][len(diag['Mdot'])//2:]),
            'ldot': lambda diag: diag['t/Ldot'][()] / diag['t/Mdot'][()]}
                            #np.mean(diag['Mdot'][len(diag['Mdot'])//2:])

def get_header_var(infile, var):
    if ',' in var:
        out = []
        for v in var.split(','):
            out.append(infile["header"][v][()])
        return out
    elif isinstance(var, list):
        out = []
        for v in var:
            out.append(infile["header"][v][()])
        return out
    else:
        return infile["header"][var][()]


def get_quiescence(infile, diag=False, set_time=None):
    if set_time is not None:
        tstart, tend = set_time
    else:
        tstart, tend = infile['avg']['start'][()], infile['avg']['end'][()]

    if diag:
        t = infile['diag']['t'][()]
    else:
        t = infile['coord']['t'][()]

    start = i_of(t, tstart)
    end = i_of(t, tend)

    return slice(start, end)


def get_result(infile, ivar, var, qui=False, only_nonzero=False, **kwargs):
    """Get the values of a variable, and of the independent variable against which to plot it.
    Various common slicing options:
    :arg qui Get only "quiescence" time, i.e. range over which time-averages were taken
    :arg only_nonzero Get only nonzero values
    """
    ret_i = get_ivar(infile, ivar, **kwargs)
    if var in infile[ivar]:
        ret_v = infile[ivar][var][()]
    elif var in fns_dict:
        ret_v = fns_dict[var](infile[ivar])
    elif var[:4] == 'log_':
        ret_i, ret_v = get_result(infile, ivar, var[4:], qui=qui, only_nonzero=only_nonzero, **kwargs)
        return ret_i, np.log10(ret_v)
    elif var in ['mdot', 'phi_b', 'ldot', 'edot']:
            return ret_i, diag_fns[var](infile)
    else:
        print("Can't find variable: {} as a function of {}".format(var, ivar))
        return None, None

    if qui:
        qui_slc = get_quiescence(infile)
        if isinstance(ret_i, list):
            ret_i = [i[qui_slc] for i in ret_i]
        else:
            ret_i = ret_i[qui_slc]
        ret_v = ret_v[qui_slc]

    if only_nonzero and len(ret_v.shape) == 1:
        nz_slc = np.nonzero(ret_v)
        if isinstance(ret_i, list):
            ret_i = [i[nz_slc] for i in ret_i]
        else:
            ret_i = ret_i[nz_slc]
        ret_v = ret_v[nz_slc]

    return ret_i, ret_v


def get_grid(infile):
    params = read_hdr(infile['header'])
    return Grid(params)


def get_ivar(infile, ivar, th_r=None, i_xy=False, mesh=True):
    """Given an input file and the string of independent variable name(s) ('r', 'rth', 'rt', etc),
    return a grid of those variables' values.
    """
    ret_i = []
    G = get_grid(infile)

    if mesh:
        native_coords = G.coord_all_mesh()
    else:
        native_coords = G.coord_all()

    if ivar[-1:] == 't':
        t = infile['coord']['t'][()]
        if mesh:
            t = np.append(t, t[-1] + (t[-1] - t[0]) / t.shape[0])
        ret_i.append(t)
    if 'r' in ivar:
        ret_i.append(G.coords.r(native_coords)[:, 0, 0])
    if 'th' in ivar:
        r1d = G.coords.r(native_coords)[:, 0, 0]
        if th_r is not None:
            th = G.coords.th(native_coords)[i_of(r1d, th_r), :, 0]
        else:
            #print("Guessing r for computing th!")
            th = G.coords.th(native_coords)[-1, :, 0]
        if 'hth' in ivar:
            th = th[:len(th)//2]
        ret_i.append(th)
    if 'phi' in ivar:
        ret_i.append(G.coords.phi(native_coords)[0, 0, :])

    # TODO handle converting 'thphi' to x-y with at_r
    # TODO handle th's r-dependence in 'rth'
    # TODO think about how to treat slices this nicely

    # Make a meshgrid of
    ret_grids = np.meshgrid(*reversed(ret_i))
    ret_grids.reverse()
    if i_xy and 'r' in ivar and 'th' in ivar:
        # XZ plot
        x = ret_grids[-2] * np.sin(ret_grids[-1])
        z = ret_grids[-2] * np.cos(ret_grids[-1])
        ret_grids[-2:] = x, z
    elif i_xy and 'r' in ivar and 'phi' in ivar:
        # XY plot
        x = ret_grids[-2] * np.cos(ret_grids[-1])
        y = ret_grids[-2] * np.sin(ret_grids[-1])
        ret_grids[-2:] = x, y

    # Squash single-variable lists for convenience
    if len(ret_grids) == 1:
        ret_grids = ret_grids[0]
        if mesh:
            # Probably no one actually wants a 1D mesh
            ret_grids = ret_grids[:-1]

    return ret_grids


def get_lc(infile, angle=163, rhigh=20, add_pol=False, qui=False):
    avg_t = get_ivar(infile, 't')
    # TODO merge lightcurves into analysis results or keep them in the same directory
    fpaths = [os.path.join(os.path.dirname(os.path.realpath(infile.filename)),
                           "{}".format(int(angle)), "m_1_1_{}".format(rhigh), "lightcurve.dat"),
              os.path.join(os.path.dirname(os.path.realpath(infile.filename)),
                           "{}".format(180-int(angle)), "m_1_1_{}".format(rhigh), "lightcurve.dat")]

    t_len = avg_t.size
    lightcurve = np.zeros(t_len)
    lightcurve_pol = np.zeros(t_len)
    for fpath in fpaths:
        if os.path.exists(fpath):
            #print("Found ", fpath)
            cols = np.loadtxt(fpath).transpose()
            # Normalize to same # elements as analysis by cheaply extending the last value for a few steps
            # TODO put this behind a default option for transparency
            f_len = cols.shape[1]
            if f_len >= t_len:
                lightcurve[:] = cols[2][:t_len]
                lightcurve_pol[:] = cols[1][:t_len]
            elif f_len < t_len:
                lightcurve[:f_len] = cols[2]
                lightcurve[f_len:] = lightcurve[f_len - 1]
                lightcurve_pol[:f_len] = cols[1]
                lightcurve_pol[f_len:] = lightcurve_pol[f_len - 1]

    #print("Polarized transport L1 difference: {}".format(np.linalg.norm(lightcurve - lightcurve_pol)))

    if qui:
        qui_slc = get_quiescence(infile)
        avg_t = avg_t[qui_slc]
        lightcurve = lightcurve[qui_slc]
        lightcurve_pol = lightcurve_pol[qui_slc]

    if add_pol:
        return avg_t, lightcurve, lightcurve_pol
    else:
        return avg_t, lightcurve


def get_diag(infile, var, only_nonzero=True, qui=False, **kwargs):
    if 'diag' in infile and 't' in infile['diag'] and var in infile['diag']:
        ret_i, ret_v = infile['diag']['t'][()], infile['diag'][var][()]

        if only_nonzero:
            slc = np.nonzero(ret_v)
            ret_i, ret_v = ret_i[slc], ret_v[slc]

        if qui:
            qui_slc = get_quiescence(infile, diag=True, **kwargs)
            ret_i, ret_v = ret_i[qui_slc], ret_v[qui_slc]

        return ret_i, ret_v
    elif var[:4] == 'log_':
        ret_i, ret_v = get_diag(infile, var[4:], only_nonzero=True, qui=False, **kwargs)
        return ret_i, np.log10(ret_v)
    else:
        return None, None

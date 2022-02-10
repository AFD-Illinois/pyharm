# Functions indexing a results directory

import os
import numpy as np
import h5py

from .grid import Grid
from .util import i_of
from .variables import fns_dict

# Specifically for reading the header as copied/output to result files
from .io.iharm3d_header import read_hdr

"""
Results are organized by remaining independent
variable -- so, a phi- and time-average will be under 'rth' since these are its remaining independent variables.

This is more logical than it sounds: this way, most quantity names contain all the info needed to plot them.
Also, multiple reductions of the same quantity can be stored with logical names, e.g. a phi,t average of rho in
'rth/rho' and a further average over th in 'r/rho'.

For variables in th/phi in FMKS, one can specify a radius r to use.
"""

window_sz = 101

def smoothed(list):
    new_list = list.copy()
    for i in range(len(list)):
        new_list[i] = np.mean(list[max(i-window_sz//2, 0):min(i+window_sz//2+1, len(list))])
    return new_list

class AnaResults(object):
    """
    Tools for dealing with the results computed by scripts/analysis.py.
    Output format documentation soon^TM but the upshot is independent variables are directories,
    dependent variables are datasets, header is in iharm3d format as written by iharm3d_header.py
    """

    diag_fns = {'mdot': lambda diag: diag['t/Mdot'][()],
                'mdot_smooth': lambda diag: smoothed(diag['t/Mdot'][()]),
                'phi_b_per': lambda diag: diag['t/Phi_b'][()] / np.sqrt(diag['t/Mdot'][()]),
                'phi_b': lambda diag: diag['t/Phi_b'][()] / np.sqrt(smoothed(diag['t/Mdot'][()])),
                'edot_per': lambda diag: diag['t/Edot'][()] / diag['t/Mdot'][()],
                'edot': lambda diag: diag['t/Edot'][()] / smoothed(diag['t/Mdot'][()]),
                'ldot_per': lambda diag: diag['t/Ldot'][()] / diag['t/Mdot'][()],
                'ldot': lambda diag: diag['t/Ldot'][()] / smoothed(diag['t/Mdot'][()]),
                }

    def __init__(self, fname):
        # Since we don't care as much about either RAM or speed here, we just read the whole thing
        self.fname = fname
        self.file = h5py.File(fname, "r")
        self.params = read_hdr(fname['/header'])
        self.grid = Grid(self.params)
        self.qui_ends = (self.file['avg']['start'][()], self.file['avg']['end'][()])
        self.qui_slc = self.get_time_slice(*self.qui_ends)
        if 'diag' in self.file:
            self.has_diag = True
            self.qui_diag = self.get_time_slice(*self.qui_ends, diag=True)

    def __del__(self):
        self.file.close()

    def get_time_slice(self, tstart, tend, diag=False):
        if diag:
            t = self.file['diag']['t'][()]
        else:
            t = self.file['coord']['t'][()]
        return slice(i_of(t, tstart), i_of(t, tend))

    def get_result(self, ivar, var, qui=False, only_nonzero=True, **kwargs):
        """Get the values of a variable, and of the independent variable against which to plot it.
        Various common slicing options:
        :arg qui Get only "quiescence" time, i.e. range over which time-averages were taken
        :arg only_nonzero Get only nonzero values
        """
        ret_i = np.array(self.get_ivar(ivar, **kwargs))
        ivar_l = ivar.replace("log_","")

        if var in self.file[ivar_l]:
            ret_v = self.file[ivar_l][var][()]
        elif var in fns_dict:
            ret_v = fns_dict[var](self.file[ivar_l])
        elif var[:4] == 'log_':
            ret_i, ret_v = self.get_result(ivar_l, var[4:], qui=qui, only_nonzero=only_nonzero, **kwargs)
            return ret_i, np.log10(ret_v)
        elif var in self.diag_fns: # TODO merge this with larger one, provide __getitem__
                return ret_i, self.diag_fns[var](self)
        else:
            print("Can't find variable: {} as a function of {}".format(var, ivar_l))
            return None, None

        if qui:
            if isinstance(ret_i, list):
                ret_i = np.array([i[self.qui_slc] for i in ret_i])
            else:
                ret_i = ret_i[self.qui_slc]
            ret_v = ret_v[self.qui_slc]

        if only_nonzero and len(ret_v.shape) == 1:
            nz_slc = np.nonzero(ret_v)
            if isinstance(ret_i, list):
                ret_i = np.array([i[nz_slc] for i in ret_i])
            else:
                ret_i = ret_i[nz_slc]
            ret_v = ret_v[nz_slc]

        return ret_i, ret_v


    def get_ivar(self, ivar, th_r=None, i_xy=False, mesh=True):
        """Given an input file and the string of independent variable name(s) ('r', 'rth', 'rt', etc),
        return a grid of those variables' values.
        """
        ret_i = []
        G = self.grid

        if mesh:
            native_coords = G.coord_all_mesh()
        else:
            native_coords = G.coord_all()

        if ivar[-1:] == 't':
            t = self.file['coord']['t'][()]
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


    def get_diag(self, var, only_nonzero=True, qui=False, **kwargs):
        if self.diag and var in self.file['diag']:
            ret_i, ret_v = self.file['diag']['t'][()], self.file['diag'][var][()]

            if only_nonzero:
                slc = np.nonzero(ret_v)
                ret_i, ret_v = ret_i[slc], ret_v[slc]

            if qui:
                qui_slc = self.get_quiescence(diag=True, **kwargs)
                ret_i, ret_v = ret_i[qui_slc], ret_v[qui_slc]

            return ret_i, ret_v
        elif var[:4] == 'log_':
            ret_i, ret_v = self.get_diag(var[4:], only_nonzero=only_nonzero, qui=qui, **kwargs)
            return ret_i, np.log10(ret_v)
        else:
            return None, None

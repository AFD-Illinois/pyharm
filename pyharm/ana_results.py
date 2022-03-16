# Functions indexing a results directory

import os
import numpy as np
import h5py

from .grid import Grid
from .util import i_of
from .variables import fns_dict

# Specifically for reading the header as copied/output to result files
from .io.iharm3d_header import read_hdr
from .io import read_log

"""
Results are organized by remaining independent variable -- so, a phi- and time-average
will be under 'rth' since these are its remaining independent variables.

This is more logical than it sounds: this way, most quantity names contain all the info
needed to plot them.  Also, multiple reductions of the same quantity can be stored with
logical names, e.g. a phi,t average of rho in 'rth/rho' and a further average over th
in 'r/rho'.

For variables in th/phi in FMKS, one can specify a radius r to use.
"""

def smoothed(a, window_sz=101):
    """A potentially slightly faster smoothing operation"""
    ret = np.array([np.mean(a[:n]) for n in range(1,window_sz//2+1)])
    ret = np.append(ret, np.convolve(a, np.ones(window_sz), 'valid') / window_sz)
    ret = np.append(ret, np.array([np.mean(a[-n:]) for n in range(1,window_sz//2+1)]))
    return ret

class AnaResults(object):
    """
    Tools for dealing with the results computed by scripts/analysis.py.
    Output format documentation soon^TM but the upshot is independent variables are directories,
    dependent variables are datasets, header is in iharm3d format as written by iharm3d_header.py
    """

    # TODO merge w/ version in variables.py?
    diag_fns = {'mdot': lambda diag: np.abs(diag['Mdot_EH']),
                'mdot_smooth': lambda diag: smoothed(diag['mdot']), # Default smoothing when plotting X/mdot normalized plots
                'phi_b_per': lambda diag: diag['Phi_b'] / np.sqrt(diag['mdot']),
                'phi_b': lambda diag: diag['Phi_b'] / np.sqrt(diag['mdot_smooth']),
                'edot_per': lambda diag: diag['Edot'] / diag['mdot'],
                'edot': lambda diag: diag['Edot'] / diag['mdot_smooth'],
                'ldot_per': lambda diag: diag['Ldot'] / diag['mdot'],
                'ldot': lambda diag: diag['Ldot'] / diag['mdot_smooth'],
                'Phi_b': lambda diag: diag['Phi_EH'],
                }

    def __init__(self, fname):
        # When reading HDF5 files, just open the file
        # When reading diagnostic output, read the whole thing
        self.fname = fname
        if ".h5" in fname or ".hdf5" in fname:
            # Read analysis results
            self.file = h5py.File(fname, "r")
            self.params = read_hdr(fname['/header'])
            self.grid = Grid(self.params)
            self.qui_ends = (self.file['avg/start'][()], self.file['avg/end'][()])
            self.qui_slc = self.get_time_slice(*self.qui_ends)
            if 'diag' in self.file:
                self.has_diag = True
                self.qui_diag = self.get_time_slice(*self.qui_ends, diag=True)
        else:
            # Read diagnostic output.  Much more limited functionality here,
            # mostly for applying diag_fns
            self.file = read_log(fname)
            self.diag_only = True
            

    def __del__(self):
        if not isinstance(self.file, dict):
            self.file.close()

    def get_time_slice(self, tstart, tend, diag=False):
        return slice(i_of(self['t_diag'], tstart), i_of(self['t_diag'], tend))

    def __getitem__(self, key):
        #print("Getting result "+key)
        if key in self.file:
            return self.file[key][()]
        elif 'coord' in self.file and key in self.file['coord']:
            return self.file['coord'][key][()]

        # Postfixes for options:
        # 
        elif key[-5:] =='_diag':
            kname = key.replace('_diag','')
            if 'diag' in self.file and kname in self.file['diag']:
                return self.file['diag'][kname][()]
            else:
                return self[kname]

        elif '_smoothed' in key:
            klist = key.split('_')
            si = klist.index('smoothed')
            try:
                window = int(klist[si+1])
                kname = '_'.join(klist[:si] + klist[si+2:])
            except (IndexError, TypeError):
                window=101
                kname = '_'.join(klist[:si] + klist[si+1:])
            return smoothed(self[kname], window_sz=window)

        # Prefixes for a few common 1:1 math operations.
        # Most math should be done by reductions.py
        # Don't bother to cache these, they aren't intensive to calculate
        elif key[:5] == "sqrt_":
            return np.sqrt(self[key[5:]])
        elif key[:4] == "abs_":
            return np.abs(self[key[4:]])
        elif key[:4] == "log_":
            return np.log10(self[key[4:]])
        elif key[:3] == "ln_":
            return np.log(self[key[3:]])

        elif key in fns_dict:
            return fns_dict[key](self)
        elif key in self.diag_fns:
            return self.diag_fns[key](self)
        elif key[:2] == "t/" and self.diag_only:
            return self[key[2:]]
        else:
            # TODO except, try other independent vars.  Both?
            return self.get_result('t', key)

    def get_result(self, ivar, var, qui=False, only_nonzero=True, **kwargs):
        """Get the values of a variable, and of the independent variable against which to plot it.
        Various common slicing options:
        :arg qui Get only "quiescence" time, i.e. range over which time-averages were taken
        :arg only_nonzero Get only nonzero values
        """
        #print("Getting result "+ivar+" "+var)
        ret_i = np.array(self.get_ivar(ivar, **kwargs))
        ivar_l = ivar.replace("log_","")

        if ivar_l in self.file and var in self.file[ivar_l]:
            ret_v = self.file[ivar_l][var][()]
        elif var in fns_dict:
            ret_v = fns_dict[var](self)
        elif var in self.diag_fns:
                return ret_i, self.diag_fns[var](self)
        else:
            raise IOError("Can't find variable: {} as a function of {}".format(var, ivar_l))

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
        """Get a grid of independent variable values
        """
        ret_i = []

        if ivar != 't':
            print(ivar)
            G = self.grid
            if mesh:
                native_coords = G.coord_all_mesh()
            else:
                native_coords = G.coord_all()

        if ivar[-1:] == 't':
            if 'coord' in self.file:
                t = self.file['coord']['t'][()]
            else:
                t = self.file['time']
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

#!/usr/bin/env python3

import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from pyHARM.ana.plot import pcolormesh_symlog
from pyHARM.ana.results import get_ivar

infile = sys.argv[1]

dirpath = os.path.join(os.path.dirname(infile), "auto_plots")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

inf_hdf5 = h5py.File(infile, "r")
#for ivar in [key for key in inf_hdf5.keys() if key not in ['avg', 'coord', 'diag', 'header', 'pdf', 'pdft']]:
for ivar in 't':
    # Split independent and dependent variable retrieval for efficiency
    ivar_d = get_ivar(infile, ivar, i_xy=True)

    for var in inf_hdf5[ivar].keys():
        try:
            fname = "{}_{}.png".format(ivar, var)
            print("Plotting {}".format(fname))

            try:
                # A bunch of custom sizes here...
                fig, ax = plt.subplots(1, 1, figsize=(12,10))
                plt.grid(True)

                var_d = inf_hdf5[ivar][var][()]
                if not isinstance(ivar_d, list):
                    if np.all(var_d >= 0):
                        if np.abs(np.max(var_d) - np.min(var_d)) < 100:
                            plt.plot(ivar_d[np.nonzero(var_d)], var_d[np.nonzero(var_d)])
                        else:
                            plt.semilogy(ivar_d[np.nonzero(var_d)], var_d[np.nonzero(var_d)])
                    else:
                        plt.plot(ivar_d, var_d)
                    # If it's a big radial plot, zoom in
                    if ivar == 'r' and ivar_d[-1] > 100:
                            plt.xlim(0, 100)
                    if ivar == 't':
                        print("Average {} over t: {}".format(var, np.mean(var_d)))
                    plt.xlabel(ivar)
                    plt.ylabel(var)
                else:
                    if np.min(var_d) >= 0:
                        # Only plot variables with any nonzero elements (hard to log-scale 0)
                        vmax = np.nanmax(var_d)
                        vmax = min(vmax, np.finfo('d').max)
                        if vmax > 0:
                            # Floor to 8 orders of magnitude
                            vmin = min(1, vmax * 1e-8)
                            var_d = np.maximum(var_d, 1.01*vmin)

                            pcm = plt.pcolormesh(*ivar_d, var_d, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap='jet')
                            plt.colorbar(pcm)
                    else:
                        pcm = pcolormesh_symlog(ax, *ivar_d, var_d)

                    if ivar[-1:] == 't':
                        plt.xlabel('t')
                        if len(ivar) > 1:
                            plt.ylabel(ivar[:-1])
                            plt.title(var)
                            #print(ivar_d.shape)
                            if 'r' in ivar and ivar_d[0][-1] > 100:
                                plt.ylim(0, 100)
                        else:
                            plt.ylabel(var)

                    elif 'r' in ivar and ('phi' in ivar or 'th' in ivar):
                        plt.xlabel('x')
                        if 'phi' in ivar:
                            plt.ylabel('y')
                            ax.set_xlim([-100, 100])
                            ax.set_ylim([-100, 100])
                        elif 'th' in ivar:
                            plt.ylabel('z')
                            if not 'hth' in ivar:
                                fig.set_size_inches((10, 15))
                                ax.set_xlim([0, 100])
                                ax.set_ylim([-100, 100])
                        plt.title(var)

                    elif ivar == 'thphi':
                        plt.xlabel('th')
                        plt.ylabel('phi')
                        plt.title(var)
                    else:
                        plt.xlabel(ivar)
                        plt.ylabel(var)
            except ValueError as e:
                print("Could not plot {}: {}".format(fname, e))
            except OverflowError as o:
                print("Could not plot {}: Overflow error".format(fname))

            fpath = os.path.join(dirpath, fname)
            plt.savefig(fpath, dpi=100)
            plt.close()
        except:
           print("Could not plot {}: {}".format(fname, sys.exc_info()[0]))

inf_hdf5.close()
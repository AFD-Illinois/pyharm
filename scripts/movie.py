

import os
import sys
import pickle
import psutil
import numpy as np
import h5py
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pyHARM
import pyHARM.io as io
from pyHARM.ana.iharm_dump import IharmDump
from pyHARM.ana.reductions import *
import pyHARM.ana.plot as bplt
import pyHARM.ana.plot_results as bpltr
from pyHARM.ana.variables import T_mixed, pretty
from pyHARM.util import i_of, calc_nthreads, run_parallel

# Movie size in inches. Keep 16/9 for standard-size movies
FIGX = 16
FIGY = 9
FIGDPI = 100

# For plotting debug, "array-space" plots
# Certain plots can override this below
USEARRSPACE = True

LOG_MDOT = False
LOG_PHI = False

# Load diagnostic data from post-processing (eht_out.p)
diag_post = False

# Default movie start & end time.
# Can be overridden on command line for splitting movies among processes
tstart = 0
tend = 1e7

# Shorthand for usual layouts:
# slc: 1 2 3 4\ 5 6 7 8 (use 1,2,5,6 maybe 7,8)
# flux: 1 2\ 3 4\ 5 6\ 7 8 (use 2,4 maybe 6,8)
ax_slc = lambda i: plt.subplot(2, 4, i)
ax_flux = lambda i: plt.subplot(4, 2, i)

def plot(n):
    tdump = io.get_dump_time(files[n])
    if (tstart is not None and tdump < tstart) or (tend is not None and tdump > tend):
        return
    
    print("frame {} / {}".format(n, len(files)-1))
    
    fig = plt.figure(figsize=(FIGX, FIGY))
    
    to_load = {}
    if "simple" not in movie_type and "floor" not in movie_type:
        # Everything but simple & pure floor movies needs derived vars
        to_load['calc_derived'] = True
    if "fail" in movie_type or movie_type == "e_ratio" or movie_type == "conservation":
        to_load['add_fails'] = True
    if "floor" in movie_type:
        to_load['add_floors'] = True
    if "current" in movie_type:
        to_load['add_jcon'] = True
    if "divB" in movie_type:
        to_load['add_divB'] = True
    # TODO U if needed

    if "_ghost" in movie_type:
        plot_ghost = True
        params = {'include_ghost': True}
    else:
        plot_ghost = False
        params = {}
        
    dump = pyHARM.load_dump(files[n], params=params, **to_load)

    # Title by time, otherwise number
    try:
        fig.suptitle("t = {}".format(dump['t']))
    except ValueError:
        fig.suptitle("dump {}".format(n))
    
    # Zoom in for small problems
    # TODO use same r1d as analysis?
    if len(dump['r'].shape) < 3:
        window = [-20, 20, -20, 20]
        nlines = 20
        rho_l, rho_h = -5, 2
    elif dump['r'][-1, 0, 0] > 100:
        window = [-25, 25, -25, 25]
        nlines = 20
        rho_l, rho_h = -5, 1
        iBZ = i_of(dump['r'][:,0,0], 100) # most MADs
        rBZ = 100
    elif dump['r'][-1, 0, 0] > 10:
        window = [-50, 50, -50, 50]
        nlines = 5
        rho_l, rho_h = -5, 2
        iBZ = i_of(dump['r'][:,0,0], 40)  # most SANEs
        rBZ = 40
    else: # Then this is a Minkowski simulation or something weird
        window = [dump['x'][0,0,0], dump['x'][-1,-1,-1], dump['y'][0,0,0], dump['y'][-1,-1,-1],]
        nlines = 0
        rho_l, rho_h = -2, 0.0
        iBZ = 1
        rBZ = 1
    
    # If we're in arrspace we definitely want a 0,1 window
    if USEARRSPACE:
        if plot_ghost:
            window = [-0.1, 1.1, -0.1, 1.1]
        else:
            window = [0, 1, 0, 1]
    
    if movie_type == "simplest_poloidal":
        # Simplest movie: just RHO, poloidal slice
        ax_slc = plt.subplot(1, 1, 1)
        var = 'log_rho'
        arrspace=False
        vmin = rho_l
        vmax = rho_h
        bplt.plot_xz(ax_slc, dump, var, label="",
                     vmin=vmin, vmax=vmax, window=window, arrayspace=arrspace,
                     xlabel=False, ylabel=False, xticks=[], yticks=[],
                     cbar=False, cmap='jet')
        plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)
    elif movie_type == "simplest_toroidal":
        # Simplest movie: just RHO, toroidal slice
        ax_slc = plt.subplot(1, 1, 1)
        var = 'log_rho'
        arrspace=False
        vmin = rho_l
        vmax = rho_h
        bplt.plot_xy(ax_slc, dump, 'log_rho', label="",
                     vmin=vmin+0.15, vmax=vmax+0.15, window=window, arrayspace=arrspace,
                     xlabel=False, ylabel=False, xticks=[], yticks=[],
                     cbar=False, cmap='jet')
        plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)
    elif movie_type == "simplest":
        # Simplest movie: just RHO
        ax_slc = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
        if dump['coordinates'] == "cartesian":
            var = 'rho'
            arrspace = True
            vmin = None
            vmax = None
        else:
            arrspace=USEARRSPACE
            # Linear version
            # var = 'rho'
            # vmin = 0
            # vmax = 1

            var = 'log_rho'
            vmin = rho_l
            vmax = rho_h
	
        bplt.plot_xz(ax_slc[0], dump, var, label="",
                     vmin=vmin, vmax=vmax, window=window, arrayspace=arrspace,
                     xlabel=False, ylabel=False, xticks=[], yticks=[],
                     cbar=False, cmap='jet')
        bplt.plot_xy(ax_slc[1], dump, var, label="",
                     vmin=vmin+0.15, vmax=vmax+0.15, window=window, arrayspace=arrspace,
                     xlabel=False, ylabel=False, xticks=[], yticks=[],
                     cbar=False, cmap='jet')
    
        pad = 0.0
        plt.subplots_adjust(hspace=0, wspace=0, left=pad, right=1 - pad, bottom=pad, top=1 - pad)
    
    elif movie_type == "simpler":
        # Simpler movie: RHO and phi
        gs = gridspec.GridSpec(2, 2, height_ratios=[6, 1], width_ratios=[16, 17])
        ax_slc = [fig.subplot(gs[0, 0]), fig.subplot(gs[0, 1])]
        ax_flux = [fig.subplot(gs[1, :])]
        bplt.plot_slices(ax_slc[0], ax_slc[1], dump, 'log_rho', vmin=rho_l, vmax=rho_h, window=window,
                         overlay_field=False, cmap='jet')
        bpltr.plot_diag(ax_flux[0], diag, 'phi_b', tline=dump['t'], logy=LOG_PHI, xlabel=False)
    elif movie_type == "simple":
        # Simple movie: RHO mdot phi
        gs = gridspec.GridSpec(3, 2, height_ratios=[4, 1, 1])
        ax_slc = [fig.subplot(gs[0, 0]), fig.subplot(gs[0, 1])]
        ax_flux = [fig.subplot(gs[1, :]), fig.subplot(gs[2, :])]
        bplt.plot_slices(ax_slc[0], ax_slc[1], dump, 'log_rho', vmin=rho_l, vmax=rho_h, window=window, cmap='jet', arrayspace=USEARRSPACE)
        bpltr.plot_diag(ax_flux[0], diag, 'Mdot', tline=dump['t'], logy=LOG_MDOT)
        bpltr.plot_diag(ax_flux[1], diag, 'Phi_b', tline=dump['t'], logy=LOG_PHI)
    
    elif movie_type == "traditional":
        ax_slc = lambda i: plt.subplot(2, 4, i)
        # Usual movie: RHO beta fluxes
        # CUTS
        bplt.plot_slices(ax_slc(1), ax_slc(2), dump, 'log_rho', label='log_rho', average=True,
                        vmin=rho_l, vmax=rho_h, cmap='jet', window=window, arrayspace=USEARRSPACE)
        bplt.plot_slices(ax_slc(3), ax_slc(4), dump, 'log_UU', label='log_UU', average=True,
                        vmin=rho_l, vmax=rho_h, cmap='jet', window=window, arrayspace=USEARRSPACE)
        bplt.plot_slices(ax_slc(5), ax_slc(6), dump, 'log_bsq', label='log_bsq', average=True,
                        vmin=rho_l, vmax=rho_h, cmap='jet', window=window, arrayspace=USEARRSPACE)
        bplt.plot_slices(ax_slc(7), ax_slc(8), dump, 'log_beta', label='log_beta', average=True,
                        vmin=rho_l, vmax=rho_h, cmap='jet', window=window, arrayspace=USEARRSPACE)
        # FLUXES
#            bpltr.plot_diag(ax_flux(2), diag, 't', 'Mdot', tline=dump['t'], logy=LOG_MDOT)
#            bpltr.plot_diag(ax_flux(4), diag, 't', 'phi_b', tline=dump['t'], logy=LOG_PHI)
        # Mixins:
        # Zoomed in RHO
#            bplt.plot_slices(ax_slc(7), ax_slc(8), dump, 'log_rho', vmin=-3, vmax=2,
#                             window=[-10, 10, -10, 10], field_overlay=False)

    elif movie_type == "vectors":
        ax_slc = lambda i: plt.subplot(2, 4, i)
        # Usual movie: RHO beta fluxes
        # CUTS
        bplt.plot_slices(ax_slc(1), ax_slc(5), dump, 'log_rho', label=pretty('log_rho'), average=True,
                        vmin=rho_l, vmax=rho_h, cmap='jet', window=window, arrayspace=USEARRSPACE)

        for i,var in zip((2,3,4,6,7,8), ("U1", "U2", "U3", "B1", "B2", "B3")):
            bplt.plot_xz(ax_slc(i), dump, np.log10(dump[var]), label=var,
                            vmin=rho_l, vmax=rho_h, cmap='Reds', window=window, arrayspace=USEARRSPACE)
            bplt.plot_xz(ax_slc(i), dump, np.log10(-dump[var]), label=var,
                            vmin=rho_l, vmax=rho_h, cmap='Blues', window=window, arrayspace=USEARRSPACE)

    elif movie_type == "vecs_cov":
        ax_slc = lambda i: plt.subplot(2, 4, i)
        for i,var in zip((1,2,3,4,5,6,7,8), ("u_0", "u_r", "u_th", "u_3","b_0", "b_r", "b_th", "b_3")):
            bplt.plot_xz(ax_slc(i), dump, np.log10(dump[var]), label=var,
                            vmin=rho_l, vmax=rho_h, cmap='Reds', window=window, arrayspace=USEARRSPACE)
            bplt.plot_xz(ax_slc(i), dump, np.log10(-dump[var]), label=var,
                            vmin=rho_l, vmax=rho_h, cmap='Blues', window=window, arrayspace=USEARRSPACE)

    elif movie_type == "vecs_con":
        ax_slc = lambda i: plt.subplot(2, 4, i)
        for i,var in zip((1,2,3,4,5,6,7,8), ("u^0", "u^r", "u^th", "u^3","b^0", "b^r", "b^th", "b^3")):
            bplt.plot_xz(ax_slc(i), dump, np.log10(dump[var]), label=var,
                            vmin=rho_l, vmax=rho_h, cmap='Reds', window=window, arrayspace=USEARRSPACE)
            bplt.plot_xz(ax_slc(i), dump, np.log10(-dump[var]), label=var,
                            vmin=rho_l, vmax=rho_h, cmap='Blues', window=window, arrayspace=USEARRSPACE)

    elif movie_type == "e_ratio":
        ax_slc = lambda i: plt.subplot(2, 4, i)
        # Energy ratios: difficult places to integrate, with failures
        bplt.plot_slices(ax_slc(1), ax_slc(2), dump, np.log10(dump['UU'] / dump['RHO']),
                            label=r"$\log_{10}(U / \rho)$", vmin=-3, vmax=3, average=True,
                            field_overlay=False, window=window, arrayspace=USEARRSPACE)
        bplt.plot_slices(ax_slc(3), ax_slc(4), dump, np.log10(dump['bsq'] / dump['RHO']),
                            label=r"$\log_{10}(b^2 / \rho)$", vmin=-3, vmax=3, average=True,
                            field_overlay=False, window=window, arrayspace=USEARRSPACE)
        bplt.plot_slices(ax_slc(5), ax_slc(6), dump, np.log10(1 / dump['beta']),
                            label=r"$\beta^{-1}$", vmin=-3, vmax=3, average=True,
                            field_overlay=False, window=window, arrayspace=USEARRSPACE)
        bplt.plot_slices(ax_slc(7), ax_slc(8), dump, (dump['fails'] != 0).astype(np.int32),
                            label="Failures", vmin=0, vmax=20, cmap='Reds', integrate=True,
                            field_overlay=False, window=window, arrayspace=USEARRSPACE)

    elif movie_type == "conservation":
        ax_slc = lambda i: plt.subplot(2, 4, i)
        ax_flux = lambda i: plt.subplot(4, 2, i)
        # Continuity plots to verify local conservation of energy, angular + linear momentum
        # Integrated T01: continuity for momentum conservation
        bplt.plot_slices(ax_slc(1), ax_slc(2), dump, T_mixed(dump, 1, 0),
                            label=r"$T^1_0$ Integrated", vmin=0, vmax=600, arrspace=True, integrate=True)
        # integrated T00: continuity plot for energy conservation
        bplt.plot_slices(ax_slc(5), ax_slc(6), dump, np.abs(T_mixed(dump, 0, 0)),
                            label=r"$T^0_0$ Integrated", vmin=0, vmax=3000, arrspace=True, integrate=True)
    
        # Usual fluxes for reference
        #bpltr.plot_diag(ax_flux[1], diag, 't', 'mdot', tline=dump['t'], logy=LOG_MDOT)

        r_out = 100

        # Radial conservation plots
        E_r = shell_sum(dump, T_mixed(dump, 0, 0)) # TODO variables
        Ang_r = shell_sum(dump, T_mixed(dump, 0, 3))
        mass_r = shell_sum(dump, dump['ucon'][0] * dump['RHO'])

        bplt.radial_plot(ax_flux(2), dump, np.abs(E_r), title='Conserved vars at R', ylim=(0, 10000), rlim=(0, r_out), label="E_r")
        bplt.radial_plot(ax_flux(2), dump, np.abs(Ang_r) / 10, ylim=(0, 10000), rlim=(0, r_out), style='r-', label="L_r")
        bplt.radial_plot(ax_flux(2), dump, np.abs(mass_r), ylim=(0, 10000), rlim=(0, r_out), style='b-', label="M_r")
        ax_flux(2).legend()
    
        # Radial energy accretion rate
        Edot_r = shell_sum(dump, T_mixed(dump, 1, 0))
        bplt.radial_plot(ax_flux(4), dump, Edot_r, label='Edot at R', ylim=(-200, 200), rlim=(0, r_out), arrayspace=True)
    
        # Radial integrated failures
        bplt.radial_plot(ax_flux(6), dump, (dump['fails'] != 0).sum(axis=(1, 2)), label='Fails at R', arrayspace=True, rlim=(0, r_out), ylim=(0, 1000))

    elif movie_type == "energies":
        ax_slc = lambda i: plt.subplot(2, 4, i)
        # Energy ratios: difficult places to integrate, with failures
        bplt.plot_slices(ax_slc(1), ax_slc(2), dump, 'log_rho',
                            label=r"$\log_{10}(U / \rho)$", vmin=-3, vmax=3, average=True,
                            field_overlay=False, window=window, arrayspace=USEARRSPACE)
        bplt.plot_slices(ax_slc(3), ax_slc(4), dump, 'log_bsq',
                            label=r"$\log_{10}(b^2 / \rho)$", vmin=-3, vmax=3, average=True,
                            field_overlay=False, window=window, arrayspace=USEARRSPACE)
        bplt.plot_slices(ax_slc(5), ax_slc(6), dump, 'log_UU',
                            label=r"$\beta^{-1}$", vmin=-3, vmax=3, average=True,
                            field_overlay=False, window=window, arrayspace=USEARRSPACE)
        bplt.plot_slices(ax_slc(7), ax_slc(8), dump, (dump['fails'] != 0).astype(np.int32),
                            label="Failures", vmin=0, vmax=20, cmap='Reds', integrate=True,
                            field_overlay=False, window=window, arrayspace=USEARRSPACE)

    elif movie_type == "floors":
        ax_slc = lambda i: plt.subplot(2, 4, i)
        bplt.plot_xz(ax_slc(1), dump, 'log_rho', label=pretty('log_rho'),
                        vmin=rho_l, vmax=rho_h, cmap='jet', window=window, arrayspace=USEARRSPACE)
        max_fail = 20
        bplt.plot_xz(ax_slc(2), dump, dump['floors'] & 1, label="GEOM_RHO",
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, arrayspace=USEARRSPACE)
        bplt.plot_xz(ax_slc(3), dump, dump['floors'] & 2, label="GEOM_U",
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, arrayspace=USEARRSPACE)
        bplt.plot_xz(ax_slc(4), dump, dump['floors'] & 4, label="B_RHO",
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, arrayspace=USEARRSPACE)
        bplt.plot_xz(ax_slc(5), dump, dump['floors'] & 8, label="B_U",
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, arrayspace=USEARRSPACE)
        bplt.plot_xz(ax_slc(6), dump, dump['floors'] & 16, label="TEMP",
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, arrayspace=USEARRSPACE)
        bplt.plot_xz(ax_slc(7), dump, dump['floors'] & 32, label="GAMMA",
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, arrayspace=USEARRSPACE)
        bplt.plot_xz(ax_slc(8), dump, dump['floors'] & 64, label="KTOT",
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, arrayspace=USEARRSPACE)

    elif movie_type == "floors_old":
        ax_slc6 = lambda i: plt.subplot(2, 3, i)
        bplt.plot_slices(ax_slc6(1), ax_slc6(2), dump, 'log_rho', label=pretty('log_rho'),
                        vmin=rho_l, vmax=rho_h, cmap='jet')
        max_fail = 1
        bplt.plot_xz(ax_slc6(3), dump, dump['floors'] == 1, label="GEOM",
                    vmin=0, vmax=max_fail, cmap='Reds')
        bplt.plot_xz(ax_slc6(4), dump, dump['floors'] == 2, label="SIGMA",
                    vmin=0, vmax=max_fail, cmap='Reds')
        bplt.plot_xz(ax_slc6(5), dump, dump['floors'] == 3, label="GAMMA",
                    vmin=0, vmax=max_fail, cmap='Reds')
        bplt.plot_xz(ax_slc6(6), dump, dump['floors'] == 4, label="KTOT",
                    vmin=0, vmax=max_fail, cmap='Reds')

    else:
        if "_ghost" in movie_type:
            l_movie_type = movie_type.replace("_ghost","")
        else:
            l_movie_type = movie_type

        # Try to make a simple movie of just the stated variable
        if not "log_" in l_movie_type:
            rho_l, rho_h = 0, 0.1
        if "_poloidal" in l_movie_type:
            ax = plt.subplot(1, 1, 1)
            var = l_movie_type[:-9]
            bplt.plot_xz(ax, dump, var, label=pretty(var),
                        vmin=rho_l, vmax=rho_h, window=window, arrayspace=USEARRSPACE,
                        cbar=True, cmap='jet')
        elif "_toroidal" in l_movie_type:
            ax = plt.subplot(1, 2, 1)
            var = l_movie_type[:-9]
            bplt.plot_xy(ax, dump, var, label=pretty(var),
                        vmin=rho_l, vmax=rho_h, window=window, arrayspace=USEARRSPACE,
                        cbar=True, cmap='jet')
        else:
            ax_slc = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
            bplt.plot_slices(ax_slc[0], ax_slc[1], dump, l_movie_type, label=pretty(l_movie_type),
                        vmin=rho_l, vmax=rho_h, window=window, arrayspace=USEARRSPACE,
                        cbar=True, cmap='jet')
        if "divB" in movie_type:
            plt.suptitle("Max divB = {}".format(np.max(dump['divB'])))

    plt.subplots_adjust(left=0.03, right=0.97)
    plt.savefig(os.path.join(frame_dir, 'frame_%08d.png' % n), dpi=FIGDPI)
    plt.close(fig)

    del dump


if __name__ == "__main__":
    # Process arguments
    if sys.argv[1] == '-d':
        debug = True
        movie_type = sys.argv[2]
        path = sys.argv[3]
        if len(sys.argv) > 4:
            tstart = float(sys.argv[4])
        if len(sys.argv) > 5:
            tend = float(sys.argv[5])
    else:
        debug = False
        movie_type = sys.argv[1]
        path = sys.argv[2]
        if len(sys.argv) > 3:
            tstart = float(sys.argv[3])
        if len(sys.argv) > 4:
            tend = float(sys.argv[4])

    # Load the file list
    files = np.sort(glob(os.path.join(path, "dump_*.h5")))
    if len(files) == 0:
        files = np.sort(glob(os.path.join(path, "*out*.phdf")))
    if len(files) == 0:
        print("INVALID PATH TO DUMP FOLDER")
        sys.exit(1)
    
    frame_dir = "frames_" + movie_type
    os.makedirs(frame_dir, exist_ok=True)
    
    # TODO diag loading
#    if movie_type not in ["simplest", "radial", "fluxes_cap", "rho_cap", "funnel_wall"]:
#        if diag_post:
#            # Load fluxes from post-analysis: more flexible
#            diag = h5py.File("eht_out.h5", 'r')
#        else:
#            # Load diagnostics from HARM itself
#            diag = io.load_log(path)
    
    if debug:
        # Run sequentially to make backtraces work
        for i in range(len(files)):
            plot(i)
    else:
        if movie_type in ["equator", "simplest"]:
            nthreads = psutil.cpu_count()
            print("Using {} threads".format(nthreads))
        else:
            nthreads = calc_nthreads(io.read_hdr(files[0]), pad=0.6)
        run_parallel(plot, len(files), nthreads)

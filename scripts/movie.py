################################################################################
#                                                                              #
#  GENERATE MOVIES FROM SIMULATION OUTPUT                                      #
#                                                                              #
################################################################################

import os
import sys
import pickle
import numpy as np
import h5py
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pyHARM.ana.iharm_dump import IharmDump
from pyHARM.h5io import get_dump_time, read_hdr
from pyHARM.ana.reductions import get_eht_disk_j_vals
import pyHARM.ana.plot as bplt
import pyHARM.ana.plot_results as bpltr
from pyHARM.ana.util import i_of, calc_nthreads, run_parallel

# Movie size in inches. Keep 16/9 for standard-size movies
FIGX = 12
FIGY = FIGX * 9 / 16

# For plotting debug, "array-space" plots
# Certain plots can override this below
USEARRSPACE = False

LOG_MDOT = False
LOG_PHI = False

# Load diagnostic data from post-processing (eht_out.p)
diag_post = True

# Default movie start & end time.
# Can be overridden on command line for splitting movies among processes
tstart = 0
tend = 1e7

def plot(n):
    imname = os.path.join(frame_dir, 'frame_%08d.png' % n)
    tdump = get_dump_time(files[n])
    if (tstart is not None and tdump < tstart) or (tend is not None and tdump > tend):
      return
    
    print("{} / {}".format((n + 1), len(files)))
    
    fig = plt.figure(figsize=(FIGX, FIGY))
    
    if movie_type not in ["simplest", "simpler", "simple"]:
      dump = IharmDump(files[n])
      # fig.suptitle("t = %d"%dump['t']) # TODO put this at the bottom somehow?
    else:
      # Simple movies don't need derived vars
      dump = IharmDump(files[n], add_jcon=False, add_derived=False)
      jmin, jmax = get_eht_disk_j_vals(dump)
    
    # Put the somewhat crazy rho values from KORAL dumps back in plottable range
    if np.max(dump['RHO']) < 1e-10:
      dump['RHO'] *= 1e15
    
    # Zoom in for small problems
    # TODO use same r1d as analysis?
    if dump['r'][-1, 0, 0] > 100:
        window = [-100, 100, -100, 100]
        nlines = 20
        rho_l, rho_h = -3, 2
        iBZ = i_of(dump['r'][:,0,0], 100)  # most MADs
        rBZ = 100
    elif dump['r'][-1, 0, 0] > 10:
        window = [-50, 50, -50, 50]
        nlines = 5
        rho_l, rho_h = -4, 1
        iBZ = i_of(dump['r'][:,0,0], 40)  # most SANEs
        rBZ = 40
    else: # Then this is a Minkowski simulation or something weird
        window = [dump['x'][0,0,0], dump['x'][-1,-1,-1], dump['y'][0,0,0], dump['y'][-1,-1,-1],]
        nlines = 0
        rho_l, rho_h = -4, 1
        iBZ = 1
        rBZ = 1
    
    if movie_type == "simplest":
        # Simplest movie: just RHO
        ax_slc = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
        if dump['metric'] == "MINKOWSKI":
            var = 'rho'
            arrspace = True
            vmin = None
            vmax = None
        else:
            var = 'log_rho'
            arrspace=False
            vmin = rho_l
            vmax = rho_h
	
        bplt.plot_xz(ax_slc[0], dump, 'log_rho', label="",
                     vmin=vmin, vmax=vmax, window=window, arrayspace=arrspace,
                     xlabel=False, ylabel=False, xticks=[], yticks=[],
                     cbar=False, cmap='jet')
        bplt.plot_xy(ax_slc[1], dump, 'log_rho', label="",
                     vmin=vmin, vmax=vmax, window=window, arrayspace=arrspace,
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
        bplt.plot_slices(ax_slc[0], ax_slc[1], dump, 'log_rho', vmin=rho_l, vmax=rho_h, window=window, cmap='jet')
        bpltr.plot_diag(ax_flux[0], diag, 'Mdot', tline=dump['t'], logy=LOG_MDOT)
        bpltr.plot_diag(ax_flux[1], diag, 'Phi_b', tline=dump['t'], logy=LOG_PHI)
    elif movie_type == "radial":
        # Just radial profiles over time
        # TODO just record these in analysis output...
        rho_r = eht_profile(dump['rho'], jmin, jmax)
        B_r = eht_profile(np.sqrt(dump['bsq']), jmin, jmax)
        uphi_r = eht_profile(dump['ucon'][:, :, :, 3], jmin, jmax)
        
        Pg = (dump['gam'] - 1.) * dump['UU']
        Pb = dump['bsq'] / 2
        
        Pg_r = partial_shell_sum(dump, Pg, jmin, jmax)
        Ptot_r = partial_shell_sum(dump, Pg + Pb, jmin, jmax)
        betainv_r = partial_shell_sum(dump, Pb / Pg, jmin, jmax)
        
        ax_slc = lambda i: plt.subplot(2, 3, i)
        bplt.radial_plot(ax_slc(1), rho_r, ylabel=r"$<\rho>$", logy=True, ylim=[1.e-2, 1.e0])
        bplt.radial_plot(ax_slc(2), Pg_r, ylabel=r"$<P_g>$", logy=True, ylim=[1.e-6, 1.e-2])
        bplt.radial_plot(ax_slc(3), B_r, ylabel=r"$<|B|>$", logy=True, ylim=[1.e-4, 1.e-1])
        bplt.radial_plot(ax_slc(4), uphi_r, ylabel=r"$<u^{\phi}>$", logy=True, ylim=[1.e-3, 1.e1])
        bplt.radial_plot(ax_slc(5), Ptot_r, ylabel=r"$<P_{tot}>$", logy=True, ylim=[1.e-6, 1.e-2])
        bplt.radial_plot(ax_slc(6), betainv_r, ylabel=r"$<\beta^{-1}>$", logy=True, ylim=[1.e-2, 1.e1])
    elif movie_type == "fluxes_cap":
        # Fluxes through a theta-phi slice at constant radius over the pole
        axes = [plt.subplot(2, 2, i) for i in range(1, 5)]
        bplt.plot_thphi(axes[0], np.log10(dump['FE'][iBZ]), iBZ, vmin=-8, vmax=-4, label=r"FE $\theta-\phi$ slice")
        bplt.plot_thphi(axes[1], np.log10(dump['FM'][iBZ]), iBZ, vmin=-8, vmax=-4, label=r"FM $\theta-\phi$ slice")
        bplt.plot_thphi(axes[2], np.log10(dump['FL'][iBZ]), iBZ, vmin=-8, vmax=-4, label=r"FL $\theta-\phi$ slice")
        bplt.plot_thphi(axes[3], np.log10(dump['RHO'][iBZ]), iBZ, vmin=-4, vmax=1, label=r"\rho $\theta-\phi$ slice")
        
        for i, axis in enumerate(axes):
            if i == 0:
                overlay_thphi_contours(axis, diag, legend=True)
            else:
                overlay_thphi_contours(axis, diag)
            max_th = dump['n2'] // 2
            x = bplt.loop_phi(dump['x'][iBZ, :max_th, :])
            y = bplt.loop_phi(dump['y'][iBZ, :max_th, :])
            prep = lambda var : bplt.loop_phi(var[:max_th, :])
            
            axis.contour(x, y, prep(dump['th'][iBZ]), [1.0], colors='k')
            axis.contour(x, y, prep(d_fns['betagamma'](dump)[iBZ]), [1.0], colors='k')
            axis.contour(x, y, prep(d_fns['sigma'](dump)[iBZ]), [1.0], colors='xkcd:green')
            axis.contour(x, y, prep(d_fns['FE'](dump)[iBZ]), [0.0], colors='xkcd:pink')
            axis.contour(x, y, prep(d_fns['Be_nob'](dump)[iBZ]), [0.02], colors='xkcd:red')
            axis.contour(x, y, prep(d_fns['mu'](dump)[iBZ]), [2.0], colors='xkcd:blue')
    
    elif movie_type == "rho_cap":
        # Note cmaps are different between left 2 and right plot, due to the latter being far away from EH
        bplt.plot_slices(plt.subplot(1, 3, 1), plt.subplot(1, 3, 2), dump, np.log10(dump['RHO']),
                         label=r"$\log_{10}(\rho)$", vmin=-3, vmax=2, cmap='jet')
        bplt.overlay_contours(plt.subplot(1, 3, 1), dump['r'], [rBZ], color='k')
        bplt.plot_thphi(plt.subplot(1, 3, 3), np.log10(dump['RHO'][iBZ, :, :]), iBZ, vmin=-4, vmax=1, label=r"$\log_{10}(\rho)$ $\theta-\phi$ slice r=" + str(rBZ))

    elif movie_type == "funnel_wall":
        rKH = 20
        iKH = i_of(dump['r'][:,0,0], rKH)
        win = [0, rBZ / 2, 0, rBZ]
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
        axes = [plt.subplot(gs[0, i]) for i in range(3)]
        bplt.plot_xz(axes[0], dump, np.log10(dump['RHO']),
                       label=r"$\log_{10}(\rho)$", vmin=-3, vmax=2, cmap='jet', window=win, shading='flat')
        
        bplt.plot_xz(axes[1], dump, np.log10(dump['ucon'][:, :, :, 3]),
                       label=r"$\log_{10}(u^{\phi})$", vmin=-3, vmax=0, cmap='Reds', window=win, cbar=False, shading='flat')
        bplt.plot_xz(axes[1], dump, np.log10(-dump['ucon'][:, :, :, 3]),
                       label=r"$\log_{10}(u^{\phi})$", vmin=-3, vmax=0, cmap='Blues', window=win, cbar=False, shading='flat')
        
        bplt.plot_xz(axes[2], dump, np.log10(dump['beta'][:, :, :, 3]),
                       label=r"$\log_{10}(u_{\phi})$", vmin=-3, vmax=3, window=win, shading='flat')
        
        for axis in axes:
            bplt.overlay_field(axis, dump, nlines=nlines * 4)
    
    elif movie_type == "kh_radii":
        if True:  # Half-theta (one jet) switch
            awindow = [0, 1, 0.5, 1]
            bwindow = [0, rBZ / 2, 0, rBZ]
        else:
            awindow = [0, 1, 0, 1]
            bwindow = [0, rBZ / 2, -rBZ / 2, rBZ / 2]
        rlevels = [10, 20, 40, 80]
        axes = [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2), plt.subplot(2, 3, 4), plt.subplot(2, 3, 5)]
        bigaxis = plt.subplot(1, 3, 3)
        for ax, rlevel in zip(axes, rlevels):
            i_r = i_of(dump['r'][:,0,0], rlevel)
            bplt.plot_thphi(ax, dump, np.log10(dump['RHO'][i_r, :, :]), i_r,
                            label=r"$\log_{10}(\rho) (r = " + str(rlevel) + ")$", vmin=-3, vmax=2, cmap='jet', shading='flat',
                            arrayspace=True, window=awindow)

        bplt.plot_xz(bigaxis, dump, np.log10(dump['ucon'][:, :, :, 3]),
                       label="", vmin=-3, vmax=0, cmap='Reds', window=bwindow, cbar=False, shading='flat')
        bplt.plot_xz(bigaxis, dump, np.log10(-dump['ucon'][:, :, :, 3]),
                       label=r"$\log_{10}(u^{\phi})$", vmin=-3, vmax=0, cmap='Blues', window=bwindow, shading='flat')
        bplt.overlay_field(bigaxis, dump)
        bplt.overlay_contours(bigaxis, dump, dump['r'][:, :, 0], levels=rlevels, color='r')
    
    else:  # All other movie types share a layout
        ax_slc = lambda i: plt.subplot(2, 4, i)
        ax_flux = lambda i: plt.subplot(4, 2, i)
        if movie_type == "traditional":
            # Usual movie: RHO beta fluxes
            # CUTS
            bplt.plot_slices(ax_slc(1), ax_slc(2), dump, 'log_rho', vmin=-3, vmax=2, cmap='jet')
            bplt.plot_slices(ax_slc(5), ax_slc(6), dump, 'log_beta', vmin=-2, vmax=2, cmap='RdBu_r')
            # FLUXES
            bpltr.plot_diag(ax_flux(2), diag, 't', 'mdot', tline=dump['t'], logy=LOG_MDOT)
            bpltr.plot_diag(ax_flux(4), diag, 't', 'phi_b', tline=dump['t'], logy=LOG_PHI)
            # Mixins:
            # Zoomed in RHO
            bplt.plot_slices(ax_slc(7), ax_slc(8), dump, 'log_rho', vmin=-3, vmax=2,
                             window=[-10, 10, -10, 10], field_overlay=False)
    
        elif movie_type == "e_ratio":
            # Energy ratios: difficult places to integrate, with failures
            bplt.plot_slices(ax_slc(0), ax_slc(1), dump, np.log10(dump['UU'] / dump['RHO']),
                             label=r"$\log_{10}(U / \rho)$", vmin=-3, vmax=3, average=True)
            bplt.plot_slices(ax_slc(2), ax_slc(3), dump, np.log10(dump['bsq'] / dump['RHO']),
                             label=r"$\log_{10}(b^2 / \rho)$", vmin=-3, vmax=3, average=True)
            bplt.plot_slices(ax_slc(4), ax_slc(5), dump, np.log10(1 / dump['beta']),
                             label=r"$\beta^{-1}$", vmin=-3, vmax=3, average=True)
            bplt.plot_slices(ax_slc(6), ax_slc(7), dump, dump['fail'] != 0,
                             label="Failures", vmin=0, vmax=20, cmap='Reds', int=True)  # , arrspace=True)
        elif movie_type == "conservation":
            # Continuity plots to verify local conservation of energy, angular + linear momentum
            # Integrated T01: continuity for momentum conservation
            bplt.plot_slices(ax_slc[0], ax_slc[1], dump, Tmixed(dump, 1, 0),
                             label=r"$T^1_0$ Integrated", vmin=0, vmax=600, arrspace=True, integrate=True)
            # integrated T00: continuity plot for energy conservation
            bplt.plot_slices(ax_slc[4], ax_slc[5], dump, np.abs(Tmixed(dump, 0, 0)),
                             label=r"$T^0_0$ Integrated", vmin=0, vmax=3000, arrspace=True, integrate=True)
        
            # Usual fluxes for reference
            bpltr.plot_diag(ax_flux[1], diag, 't', 'mdot', tline=dump['t'], logy=LOG_MDOT)

            # Radial conservation plots
            E_r = sum_shell(dump, Tmixed(dump, 0, 0)) # TODO variables
            Ang_r = sum_shell(dump, Tmixed(dump, 0, 3))
            mass_r = sum_shell(dump['ucon'][:, :, :, 0] * dump['RHO'])
        
            # TODO switch to diag_plot as I think I record all these
            bplt.radial_plot(ax_flux[3], dump, np.abs(E_r), 'Conserved vars at R', ylim=(0, 1000), rlim=(0, 20), arrayspace=True)
            bplt.radial_plot(ax_flux[3], dump, np.abs(Ang_r) / 10, '', ylim=(0, 1000), rlim=(0, 20), col='r', arrayspace=True)
            bplt.radial_plot(ax_flux[3], dump, np.abs(mass_r), '', ylim=(0, 1000), rlim=(0, 20), col='b', arrayspace=True)
        
            # Radial energy accretion rate
            Edot_r = sum_shell(dump, Tmixed(dump, dump, 1, 0))
            bplt.radial_plot(ax_flux[5], dump, np.abs(Edot_r), 'Edot at R', ylim=(0, 200), rlim=(0, 20), arrayspace=True)
        
            # Radial integrated failures
            bplt.radial_plot(ax_flux[7], dump, (dump['fail'] != 0).sum(axis=(1, 2)), 'Fails at R', arrayspace=True, rlim=[0, 50], ylim=[0, 1000])
    
        elif movie_type == "floors":
            # TODO add measures of all floors' efficacy.  Record ceilings in header or extras?
            bplt.plot_slices(ax_flux[0], ax_flux[1], dump, dump['sigma'] - 100,
                             vmin=-100, vmax=100, cmap='RdBu_r')
            bpltr.plot_diag(ax, diag, 't', 'sigma_max', tline=dump['t'])
    
        elif movie_type in d_fns:  # Hail mary for plotting new functions one at a time
            axes = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
            win = [l * 2 for l in window]
            var = d_fns[movie_type](dump)
            bplt.plot_slices(axes[0], axes[1], dump, np.log10(var), vmin=-3, vmax=3, cmap='Reds', window=win)
            bplt.plot_slices(axes[0], axes[1], dump, np.log10(-var), vmin=-3, vmax=3, cmap='Blues', window=win)
        else:
            print("Movie type not known!")
            exit(1)
    
        # Extra padding for crowded 4x2 plots
        pad = 0.03
        plt.subplots_adjust(left=pad, right=1 - pad, bottom=pad, top=1 - pad)
    
    plt.savefig(imname, dpi=1920 / FIGX)  # TODO the group projector is like 4:3 man
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
    
    # LOAD FILES
    files = np.sort([file for file in glob(os.path.join(path, "*.h5"))
                     if "grid" not in file and "eht" not in file])
    if len(files) == 0:
        print("INVALID PATH TO DUMP FOLDER")
        sys.exit(1)
    
    frame_dir = "frames_" + movie_type
    os.makedirs(frame_dir, exist_ok=True)
    
    if movie_type not in ["simplest", "radial", "fluxes_cap", "rho_cap", "funnel_wall"]:
        if diag_post:
            # Load fluxes from post-analysis: more flexible
            diag = h5py.File("eht_out.h5", 'r')
        else:
            # Load diagnostics from HARM itself
            diag = io.load_log(path)
    
    if debug:
        # Run sequentially to make backtraces work
        for i in range(len(files)):
            plot(i)
    else:
        nthreads = calc_nthreads(read_hdr(files[0]), pad=0.3)
        run_parallel(plot, len(files), nthreads)

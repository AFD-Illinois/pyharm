# Various full figures, combining plots & settings frequently used together
# Similar to reports.py in imtools

import os
import sys
import psutil
import numpy as np
import matplotlib.pyplot as plt

from .plots import *
from ..defs import FloorFlag_KHARMA, FloorFlag_iharm3d

"""Various full figures, combining plots & settings frequently used together
Similar to reports.py in imtools"""

def simplest(dump, type="both", var='log_rho', vmin=None, vmax=None, window=(-50,50,-50,50)):
    # Simplest movie: just RHO
    if type == "both":
        fig, ax_slc = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
    else:
        fig, ax_slc = plt.subplot(1, 1, 1)

    if type in ("poloidal", "both"):
        plot_xz(ax_slc, dump, var, label="",
                vmin=None, vmax=vmax, window=window,
                xlabel=False, ylabel=False, xticks=[], yticks=[],
                cbar=False, cmap='jet')
    if type in ("toroidal", "both"):
        plot_xy(ax_slc, dump, var, label="",
                vmin=vmin+0.15, vmax=vmax+0.15, window=window,
                xlabel=False, ylabel=False, xticks=[], yticks=[],
                cbar=False, cmap='jet')
    ax_slc.axis('off')
    plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)
    return fig


def simpler(dump, type="both", var='log_rho', vmin=None, vmax=None, window=(-50,50,-50,50)):
    """Simpler movie: RHO and phi"""
    gs = gridspec.GridSpec(2, 2, height_ratios=[6, 1], width_ratios=[16, 17])
    ax_slc = [fig.subplot(gs[0, 0]), fig.subplot(gs[0, 1])]
    ax_flux = [fig.subplot(gs[1, :])]
    plot_slices(ax_slc[0], ax_slc[1], dump, 'log_rho', vmin=rho_l, vmax=rho_h, window=window,
                        overlay_field=False, cmap='jet')
    ppltr.plot_diag(ax_flux[0], diag, 'phi_b', tline=dump['t'], logy=LOG_PHI, xlabel=False)

def simple(dump, type="both", var='log_rho', vmin=None, vmax=None, window=(-50,50,-50,50)):
    """Simple movie: RHO mdot phi"""
    gs = gridspec.GridSpec(3, 2, height_ratios=[4, 1, 1])
    ax_slc = [fig.subplot(gs[0, 0]), fig.subplot(gs[0, 1])]
    ax_flux = [fig.subplot(gs[1, :]), fig.subplot(gs[2, :])]
    plot_slices(ax_slc[0], ax_slc[1], dump, 'log_rho', vmin=rho_l, vmax=rho_h, window=window, cmap='jet', native=USEARRSPACE)
    ppltr.plot_diag(ax_flux[0], diag, 'Mdot', tline=dump['t'], logy=LOG_MDOT)
    ppltr.plot_diag(ax_flux[1], diag, 'Phi_b', tline=dump['t'], logy=LOG_PHI)

def traditional(dump, type="both", vmin=None, vmax=None, window=(-50,50,-50,50)):
    ax_slc = lambda i: plt.subplot(2, 4, i)
    # Usual movie: RHO beta fluxes
    # CUTS
    plot_slices(ax_slc(1), ax_slc(2), dump, 'log_rho', label='log_rho', average=True,
                    vmin=rho_l, vmax=rho_h, cmap='jet', window=window, native=USEARRSPACE)
    plot_slices(ax_slc(3), ax_slc(4), dump, 'log_UU', label='log_UU', average=True,
                    vmin=rho_l, vmax=rho_h, cmap='jet', window=window, native=USEARRSPACE)
    plot_slices(ax_slc(5), ax_slc(6), dump, 'log_bsq', label='log_bsq', average=True,
                    vmin=rho_l, vmax=rho_h, cmap='jet', window=window, native=USEARRSPACE)
    plot_slices(ax_slc(7), ax_slc(8), dump, 'log_beta', label='log_beta', average=True,
                    vmin=rho_l, vmax=rho_h, cmap='jet', window=window, native=USEARRSPACE)
    # FLUXES
#            ppltr.plot_diag(ax_flux(2), diag, 't', 'Mdot', tline=dump['t'], logy=LOG_MDOT)
#            ppltr.plot_diag(ax_flux(4), diag, 't', 'phi_b', tline=dump['t'], logy=LOG_PHI)
    # Mixins:
    # Zoomed in RHO
#            plot_slices(ax_slc(7), ax_slc(8), dump, 'log_rho', vmin=-3, vmax=2,
#                             window=[-10, 10, -10, 10], field_overlay=False)


def prims(dump, type="poloidal", vmin=None, vmax=None, window=(-50,50,-50,50)):
    ax_slc = lambda i: plt.subplot(2, 4, i)
    if type == "poloidal":
        fn = plot_xz
    else:
        fn = plot_xy
    for i,var in enumerate(['RHO', 'UU', 'U1', 'U2', 'U3', 'B1', 'B2', 'B3']):
        fn(ax_slc(i+1), dump, var, label="",
                vmin=vmin, vmax=vmax, window=window,
                xlabel=False, ylabel=False, xticks=[], yticks=[],
                cbar=False, cmap='jet')

def vecs_prim(dump, vmin=None, vmax=None, window=(-50,50,-50,50)):
    ax_slc = lambda i: plt.subplot(2, 4, i)
    # Usual movie: RHO beta fluxes
    # CUTS
    plot_slices(ax_slc(1), ax_slc(5), dump, 'log_rho', label=pretty('log_rho'), average=True,
                    vmin=rho_l, vmax=rho_h, cmap='jet', window=window, native=USEARRSPACE)

    for i,var in zip((2,3,4,6,7,8), ("U1", "U2", "U3", "B1", "B2", "B3")):
        plot_xz(ax_slc(i), dump, np.log10(dump[var]), label=var,
                        vmin=rho_l, vmax=rho_h, cmap='Reds', window=window, native=USEARRSPACE)
        plot_xz(ax_slc(i), dump, np.log10(-dump[var]), label=var,
                        vmin=rho_l, vmax=rho_h, cmap='Blues', window=window, native=USEARRSPACE)

def vecs_cov(dump, vmin=None, vmax=None, window=(-50,50,-50,50)):
    ax_slc = lambda i: plt.subplot(2, 4, i)
    for i,var in zip((1,2,3,4,5,6,7,8), ("u_0", "u_r", "u_th", "u_3","b_0", "b_r", "b_th", "b_3")):
        plot_xz(ax_slc(i), dump, np.log10(dump[var]), label=var,
                        vmin=rho_l, vmax=rho_h, cmap='Reds', window=window, native=USEARRSPACE)
        plot_xz(ax_slc(i), dump, np.log10(-dump[var]), label=var,
                        vmin=rho_l, vmax=rho_h, cmap='Blues', window=window, native=USEARRSPACE)

def vecs_con(dump, vmin=None, vmax=None, window=(-50,50,-50,50)):
    ax_slc = lambda i: plt.subplot(2, 4, i)
    for i,var in zip((1,2,3,4,5,6,7,8), ("u^0", "u^r", "u^th", "u^3","b^0", "b^r", "b^th", "b^3")):
        plot_xz(ax_slc(i), dump, np.log10(dump[var]), label=var,
                        vmin=rho_l, vmax=rho_h, cmap='Reds', window=window, native=USEARRSPACE)
        plot_xz(ax_slc(i), dump, np.log10(-dump[var]), label=var,
                        vmin=rho_l, vmax=rho_h, cmap='Blues', window=window, native=USEARRSPACE)

def ejection(dump, vmin=None, vmax=None, window=(-50,50,-50,50)):
    ax_slc = lambda i: plt.subplot(1, 2, i)
    # Usual movie: RHO beta fluxes
    # CUTS
    plot_xz(ax_slc(1), dump, 'log_rho', label=pretty('log_rho')+" phi-average", average=True,
                    vmin=rho_l, vmax=rho_h, cmap='jet', window=window, native=USEARRSPACE)
    plot_xz(ax_slc(2), dump, 'log_bsq', label=pretty('log_bsq')+" phi-average", average=True,
                    vmin=rho_l, vmax=rho_h, cmap='jet', window=window, native=USEARRSPACE)

def b_bug(dump, vmin=None, vmax=None, window=(-50,50,-50,50)):
    rmax = 10
    thmax = 10
    phi = 100
    ax_slc = lambda i: plt.subplot(1, 3, i)
    ax_slc(1).pcolormesh(dump['X1'][:rmax,0:thmax,phi], dump['X2'][:rmax,0:thmax,phi], dump['log_b^r'][:rmax,0:thmax,phi], vmax=0, vmin=-4)
    ax_slc(2).pcolormesh(dump['X1'][:rmax,0:thmax,phi], dump['X2'][:rmax,0:thmax,phi], dump['log_b^th'][:rmax,0:thmax,phi], vmax=0, vmin=-4)
    ax_slc(3).pcolormesh(dump['X1'][:rmax,0:thmax,phi], dump['X2'][:rmax,0:thmax,phi], dump['log_b^3'][:rmax,0:thmax,phi], vmax=0, vmin=-4)

def e_ratio(dump, vmin=None, vmax=None, window=(-50,50,-50,50)):
    ax_slc = lambda i: plt.subplot(2, 4, i)
    # Energy ratios: difficult places to integrate, with failures
    plot_slices(ax_slc(1), ax_slc(2), dump, np.log10(dump['UU'] / dump['RHO']),
                        label=r"$\log_{10}(U / \rho)$", vmin=-3, vmax=3, average=True,
                        field_overlay=False, window=window, native=USEARRSPACE)
    plot_slices(ax_slc(3), ax_slc(4), dump, np.log10(dump['bsq'] / dump['RHO']),
                        label=r"$\log_{10}(b^2 / \rho)$", vmin=-3, vmax=3, average=True,
                        field_overlay=False, window=window, native=USEARRSPACE)
    plot_slices(ax_slc(5), ax_slc(6), dump, np.log10(1 / dump['beta']),
                        label=r"$\beta^{-1}$", vmin=-3, vmax=3, average=True,
                        field_overlay=False, window=window, native=USEARRSPACE)
    plot_slices(ax_slc(7), ax_slc(8), dump, (dump['fails'] != 0).astype(np.int32),
                        label="Failures", vmin=0, vmax=20, cmap='Reds', integrate=True,
                        field_overlay=False, window=window, native=USEARRSPACE)

def e_ratio_funnel(dump, vmin=None, vmax=None, window=(-50,50,-50,50)):
    ax_slc = lambda i: plt.subplot(2, 4, i)
    # Energy ratios: difficult places to integrate, with failures
    r_i = i_of(r1d, float(movie_type.split("_")[-1]))
    plot_thphi(ax_slc(1), dump, np.log10(dump['UU'] / dump['RHO']), r_i,
                        label=r"$\log_{10}(U / \rho)$", vmin=-3, vmax=3, average=True,
                        field_overlay=False, window=window, native=USEARRSPACE)
    plot_thphi(ax_slc(2), dump, np.log10(dump['UU'] / dump['RHO']), r_i,
                        label=r"$\log_{10}(U / \rho)$", vmin=-3, vmax=3, average=True,
                        field_overlay=False, window=window, native=USEARRSPACE)
    plot_thphi(ax_slc(3), dump, np.log10(dump['bsq'] / dump['RHO']), r_i,
                        label=r"$\log_{10}(b^2 / \rho)$", vmin=-3, vmax=3, average=True,
                        field_overlay=False, window=window, native=USEARRSPACE)
    plot_thphi(ax_slc(4), dump, np.log10(dump['bsq'] / dump['RHO']), r_i,
                        label=r"$\log_{10}(b^2 / \rho)$", vmin=-3, vmax=3, average=True,
                        field_overlay=False, window=window, native=USEARRSPACE)
    plot_thphi(ax_slc(5), dump, np.log10(1 / dump['beta']), r_i,
                        label=r"$\beta^{-1}$", vmin=-3, vmax=3, average=True,
                        field_overlay=False, window=window, native=USEARRSPACE)
    plot_thphi(ax_slc(6), dump, np.log10(1 / dump['beta']), r_i,
                        label=r"$\beta^{-1}$", vmin=-3, vmax=3, average=True,
                        field_overlay=False, window=window, native=USEARRSPACE)
    plot_thphi(ax_slc(7), dump, (dump['fails'] != 0).astype(np.int32), r_i,
                        label="Failures", vmin=0, vmax=20, cmap='Reds', integrate=True,
                        field_overlay=False, window=window, native=USEARRSPACE)
    plot_thphi(ax_slc(8), dump, (dump['fails'] != 0).astype(np.int32), r_i,
                        label="Failures", vmin=0, vmax=20, cmap='Reds', integrate=True,
                        field_overlay=False, window=window, native=USEARRSPACE)

def conservation(dump, vmin=None, vmax=None, window=(-50,50,-50,50)):
    ax_slc = lambda i: plt.subplot(2, 4, i)
    ax_flux = lambda i: plt.subplot(4, 2, i)
    # Continuity plots to verify local conservation of energy, angular + linear momentum
    # Integrated T01: continuity for momentum conservation

    plot_slices(ax_slc(1), ax_slc(2), dump, T_mixed(dump, 1, 0),
                        label=r"$T^1_0$ Integrated", vmin=0, vmax=2000, arrspace=True, integrate=True)
    # integrated T00: continuity plot for energy conservation
    plot_slices(ax_slc(5), ax_slc(6), dump, np.abs(T_mixed(dump, 0, 0)),
                        label=r"$T^0_0$ Integrated", vmin=0, vmax=3000, arrspace=True, integrate=True)

    # Usual fluxes for reference
    #ppltr.plot_diag(ax_flux[1], diag, 't', 'mdot', tline=dump['t'], logy=LOG_MDOT)

    r_out = 100

    # Radial conservation plots
    E_r = shell_sum(dump, T_mixed(dump, 0, 0)) # TODO variables
    Ang_r = shell_sum(dump, T_mixed(dump, 0, 3))
    mass_r = shell_sum(dump, dump['ucon'][0] * dump['RHO'])

    max_e = 50000
    radial_plot(ax_flux(2), dump, np.abs(E_r), title='Conserved vars at R', ylim=(0, max_e), rlim=(0, r_out), label="E_r")
    radial_plot(ax_flux(2), dump, np.abs(Ang_r) / 10, ylim=(0, max_e), rlim=(0, r_out), color='r', label="L_r")
    radial_plot(ax_flux(2), dump, np.abs(mass_r), ylim=(0, max_e), rlim=(0, r_out), color='b', label="M_r")
    ax_flux(2).legend()

    # Radial energy accretion rate
    Edot_r = shell_sum(dump, T_mixed(dump, 1, 0))
    radial_plot(ax_flux(4), dump, Edot_r, label='Edot at R', ylim=(-200, 200), rlim=(0, r_out), native=True)

    # Radial integrated failures
    radial_plot(ax_flux(6), dump, (dump['fails'] != 0).sum(axis=(1, 2)), label='Fails at R', native=True, rlim=(0, r_out), ylim=(0, 1000))

def energies(dump, vmin=None, vmax=None, window=(-50,50,-50,50)):
    ax_slc = lambda i: plt.subplot(2, 4, i)
    # Energy ratios: difficult places to integrate, with failures
    plot_slices(ax_slc(1), ax_slc(2), dump, 'log_rho',
                        label=r"$\log_{10}(U / \rho)$", vmin=-3, vmax=3, average=True,
                        field_overlay=False, window=window, native=USEARRSPACE)
    plot_slices(ax_slc(3), ax_slc(4), dump, 'log_bsq',
                        label=r"$\log_{10}(b^2 / \rho)$", vmin=-3, vmax=3, average=True,
                        field_overlay=False, window=window, native=USEARRSPACE)
    plot_slices(ax_slc(5), ax_slc(6), dump, 'log_UU',
                        label=r"$\beta^{-1}$", vmin=-3, vmax=3, average=True,
                        field_overlay=False, window=window, native=USEARRSPACE)
    plot_slices(ax_slc(7), ax_slc(8), dump, (dump['fails'] != 0).astype(np.int32),
                        label="Failures", vmin=0, vmax=20, cmap='Reds', integrate=True,
                        field_overlay=False, window=window, native=USEARRSPACE)

def floors(dump, vmin=None, vmax=None, window=(-50,50,-50,50)):
    ax_slc = lambda i: plt.subplot(2, 4, i)
    plot_xz(ax_slc(1), dump, 'log_rho', label=pretty('log_rho'),
                    vmin=None, vmax=None, cmap='jet', window=window, native=USEARRSPACE)
    max_fail = 20
    for i,ff in enumerate(FloorFlag_KHARMA):
        plot_xz(ax_slc(2+i), dump, dump['floors'] & ff.value, label=ff.name,
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, native=USEARRSPACE)

def old_floors():
    ax_slc6 = lambda i: plt.subplot(2, 3, i)
    plot_xz(ax_slc(1), dump, 'log_rho', label=pretty('log_rho'),
                    vmin=None, vmax=None, cmap='jet', window=window, native=USEARRSPACE)
    max_fail = 20
    for i,ff in enumerate(FloorFlag_KHARMA):
        plot_xz(ax_slc(2+i), dump, dump['floors'] & ff.value, label=ff.name,
                    vmin=0, vmax=max_fail, cmap='Reds', integrate=True, window=window, native=USEARRSPACE)


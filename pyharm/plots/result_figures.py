__license__ = """
 File: result_figures.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020-2026, pyharm contributors
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import matplotlib.pyplot as plt

import pyharm
from pyharm.defs import Loci
from pyharm.util import i_of

__doc__ = \
"""Plots of post-reduction results as computed by pyharm-analysis & read by ana_results.py.
WIP.
"""

max_this_invocation = {}

# def _get_t_slice(result, arange):
#     """Returns a time slice corresponding to the tuple or number 'arange'
#     (optionally negative-indexed from sim end)
#     """
#     # TODO BOUNDS CORRECTLY
#     if isinstance(arange, slice) or isinstance(arange, tuple) or isinstance(arange, list):
#         try:
#             return result.get_time_slice(*arange)
#         except KeyError:
#             return None
#     elif arange is not None:
#         # Min only, negative offset from end accepted
#         try:
#             return result.get_time_slice(arange)
#         except KeyError:
#             return None
#     else:
#         return True, slice(None)

def _get_r_slice(result, rrange):
    """Get a slice of radial zones matching the plot window.
    Plotting only the necessary radial range makes auto-scaling of the y-axis work
    """
    return slice(max(i_of(result['r'], rrange[0]), 0), i_of(result['r'], rrange[1]))

def _trunc_at_spin(tag):
    model_lst = tag.split(" ")
    model_lst_trunc = []
    for m in model_lst:
        if " a" in m or "$a" in m:
            break
        model_lst_trunc.append(m)
    return " ".join(model_lst_trunc)

def _model_pretty(folder):
    model = folder.split("/")
    if len(model) >= 2:
        if "_" in model[-1]:
            return model[-3]
        return model[-2].upper()+r" $"+model[-1]+r"^\circ$"
    else:
        return folder.replace("")

def _radial_profile(ax, result, var, **kwargs):

    # Get the indices/slice to average
    avg_slice = result.get_time_slice(*kwargs['arange'])
    # if not isinstance(avg_slice, slice):
    #     avg_slice = avg_slice[1]

    # Get just the relevant radial slice so y-limits get set properly
    window = (kwargs['xmin'] if kwargs['xmin'] is not None else 1,
              kwargs['xmax'] if kwargs['xmax'] is not None else 500)
    r_slice = _get_r_slice(result, (window[0]-1, window[1]+20))

    # TODO warn on nans
    tyvals = np.nan_to_num(result['rt/{}'.format(var)][avg_slice, r_slice])
    yvals = np.mean(tyvals, axis=0)

    if "Area" in var or True:
        #yvals /= yvals[i_of(result['r'], 100)]
        if np.max(yvals) > max_this_invocation[var]:
            max_this_invocation[var] = np.max(yvals)
        kwargs['ymax'] = 2.*max_this_invocation[var]

    p = ax.plot(result['r'][r_slice], yvals, label=result.tag)
    if kwargs['plot_std']:
        yerrs = np.std(tyvals, axis=0)
        ax.fill_between(result['r'][r_slice], yvals-yerrs, yvals+yerrs, alpha=0.5, color=p[0].get_color())

    if kwargs['plot_eh']:
        ax.axvline(diag['r_eh'], color='k')
    else:
        ax.set_xlim(window[0], window[1])

    # TODO only if ymin_prop?
    # We set this according to the max, it might be nan/inf
    if np.isfinite(kwargs['ymax']):
        if kwargs['ymin_prop'] is None:
            kwargs['ymin_prop'] = 1e-5
        if not "beta" in var:
            ax.set_ylim((kwargs['ymax']*kwargs['ymin_prop'], np.abs(kwargs['ymax'])))
        else: # TODO ymin_prop_beta
            ax.set_ylim((kwargs['ymax']*kwargs['ymin_prop']*kwargs['ymin_prop'], kwargs['ymax']))

    # TODO as defaults instead
    #if kwargs['logy']:
    ax.set_yscale('log')
    #if kwargs['logx']:
    ax.set_xscale('log')

    ax.set_xlabel(r"Radius [$r_g$]")
    ax.set_ylabel(pyharm.pretty(var)) #, rotation=0, ha='right')
    ax.grid(True)

def _plot_radial_averages(results, kwargs, vars, max_nx=4):
    global max_this_invocation
    for var in vars:
        if not var in max_this_invocation:
            max_this_invocation[var] = 0.

    # Radial profiles of variables
    nx = min(len(vars), max_nx)
    ny = (len(vars) - 1) // max_nx + 1
    fig, _ = plt.subplots(ny, nx, figsize=(4*nx+1,4*ny))
    ax = fig.get_axes()
    for result in results:
        for a,var in enumerate(vars):
            window = _radial_profile(ax[a], result, var, **kwargs)

    if kwargs['fig_wspace'] is None:
        kwargs['fig_wspace'] = 0.4
    return fig

def radial_profiles(results, kwargs):
    return _plot_radial_averages(results, kwargs, vars=('rho', 'Pg', 'b', 'bsq', 'Ptot', 'u^phi', 'beta_post', 'sigma_post'))

def disk_radial_profiles(results, kwargs):
    return _plot_radial_averages(results, kwargs, vars=('rho_disk', 'Pg_disk', 'b_disk', 'bsq_disk',
                                'Ptot_disk', 'u^phi_disk', 'beta_post_disk', 'sigma_post_disk'))

def notdisk_radial_profiles(results, kwargs):
    return _plot_radial_averages(results, kwargs, vars=('rho_notdisk', 'Pg_notdisk', 'b_notdisk', 'bsq_notdisk',
                                'Ptot_notdisk', 'u^phi_notdisk', 'beta_post_notdisk', 'sigma_post_notdisk'))

def jet_radial_profiles(results, kwargs):
    # 'b^r_jet', 'b^th_jet', 'b^phi_jet',
    return _plot_radial_averages(results, kwargs, vars=('rho_jet', 'Pg_jet', 'u^r_jet', 'u^th_jet', 'u^phi_jet', 'b_jet', 'inv_beta_jet', 'Ptot_jet'))


def radial_fluxes(results, kwargs):
    if kwargs['ymin_prop'] is None:
        kwargs['ymin_prop'] = 1.e-2
    return _plot_radial_averages(results, kwargs, vars=('FE_all', 'FM_all', 'FL_all'))

def disk_radial_fluxes(results, kwargs):
    if kwargs['ymin_prop'] is None:
        kwargs['ymin_prop'] = 1.e-2
    return _plot_radial_averages(results, kwargs, vars=('FE_disk', 'FM_disk', 'FL_disk'))

def notdisk_radial_fluxes(results, kwargs):
    if kwargs['ymin_prop'] is None:
        kwargs['ymin_prop'] = 1.e-2
    return _plot_radial_averages(results, kwargs, vars=('FE_notdisk', 'FM_notdisk', 'FL_notdisk'))

def jet_radial_fluxes(results, kwargs):
    # All fluxes defined positive-out -> positive where it matters, by default
    if kwargs['ymin_prop'] is None:
        kwargs['ymin_prop'] = 1.e-2
    return _plot_radial_averages(results, kwargs, vars=('Mdot_jet', 'P_jet', 'sqrt_Area_jet', 'P_EM_jet', 'P_PAKE_jet', 'P_EN_jet',), max_nx=3)


def disk_momentum_profile(results, kwargs):
    return _plot_radial_averages(results, kwargs, vars=('u_phi_disk',))

def disk_velocity_profile(results, kwargs):
    return _plot_radial_averages(results, kwargs, vars=('u^phi_disk',))


def _hth_profile(ax, result, var, arange=-1000, print_time=False, plot_std=False, ylim=None):

    # Get the times to average
    avg_slice = result.get_time_slice(*arange)
    if len(np.squeeze(result['t'][avg_slice]).shape) == 0:
        return None
    times = (round(np.squeeze(result['t'][avg_slice])[0]/1000)*1000,
             round(np.squeeze(result['t'][avg_slice])[-1]/1000)*1000)

    tyvals = result['htht/{}'.format(var)][avg_slice, :]

    yvals = np.mean(tyvals, axis=0)
    p = ax.plot(result['hth'], yvals, label=result.tag)
    if plot_std:
        yerrs = np.std(tyvals, axis=0)
        ax.fill_between(result['hth'], yvals-yerrs, yvals+yerrs, alpha=0.5, color=p[0].get_color())

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(pyharm.pretty(var), rotation=0, ha='right')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True)

def _plot_hth_profiles(results, kwargs, vars, ylim=None):
    # Radial profiles of variables
    nx = min(len(vars), 4)
    ny = (len(vars) - 1) // 4 + 1
    fig, _ = plt.subplots(ny, nx, figsize=(5*nx+4,5*ny))
    ax = fig.get_axes()
    for result in results:
        for a,var in enumerate(vars):
            window = _hth_profile(ax[a], result, var, ylim=ylim)

    if kwargs['fig_right'] is None:
        kwargs['fig_right'] = 0.6
    return fig

def omega_bz(results, kwargs):
    return _plot_hth_profiles(results, kwargs, ('omega_rel',), ylim=(0, 1))

def omega_bz_std(results, kwargs):
    return _plot_hth_profiles(results, kwargs, ('omega_rel',), plot_std=True, ylim=(0, 1))

# TODO all the BZ types comparisons

def _plot_time_evolution(ax, result, var, arange=None, show_arange=True, label=None,
                         logy=False, verbose=False, stats=False):
    data = result[f't/{var}']

    # TODO special-case somewhere else
    if 'phi_b' in var:
        data *= np.sqrt(4*np.pi)

    if result.prefer_hst:
        time = result['diag/time']
    else:
        time = result['t']
    
    global max_this_invocation
    if not 'tmin' in max_this_invocation or time[0] < max_this_invocation['tmin']:
        max_this_invocation['tmin'] = time[0]
    if not 'tmax' in max_this_invocation or time[-1] > max_this_invocation['tmax']:
        max_this_invocation['tmax'] = time[-1]

    if verbose:
        print(f"Plotting {var}: {data.shape} vs t: {time.shape}")

    if label is None:
        label = result.tag

    pt = ax.plot(np.squeeze(time), np.squeeze(data), label=label)

    if logy:
        if np.any(data < 0.):
            ax.set_yscale('symlog', linthresh=1e-1) # TODO configurable
        else:
            ax.set_yscale('log')

    if stats:
        print(f"{result.tag} peak {var}: {np.max(data)} at {np.squeeze(time)[np.argmax(np.squeeze(data))]}")

    if arange is not None:
        # Get the times to average
        avg_slice = result.get_time_slice(*arange)
        times = (round(time[avg_slice][0]/1000)*1000,
                round(time[avg_slice][-1]/1000)*1000)
        avg = np.mean(data[avg_slice])
        if show_arange:
            ax.hlines(avg, times[0], times[1], colors=pt[0].get_color(), linestyles='dashed')
            ax.text(times[1], avg, f"{avg:.2f}")
        if stats:
            print(f"{result.tag} avg {var}, {times[0]}-{times[1]}: {avg}")



    # TODO only tilt long names
    ax.set_ylabel(pyharm.pretty(var), rotation=0, ha='right')
    ax.grid(True)

def _plot_time_evolutions(results, kwargs, vars, fig=None):

    # Default spacing
    if kwargs['fig_left'] is None:
        kwargs['fig_left'] = 0.22

    if kwargs['per']:
        vars = [v+"_per" for v in vars]
    xsize = float(kwargs['fig_x']) if kwargs['fig_x'] is not None else 5
    ysize = float(kwargs['fig_y']) if kwargs['fig_y'] is not None else 1.5*len(vars)
    if fig is None:
        fig, _ = plt.subplots(len(vars), 1, figsize=(xsize, ysize))
    ax = fig.get_axes()

    for result in results:
        arange = kwargs['arange'] if 'arange' in kwargs else None
        for a,var in enumerate(vars):
            # TODO passthrough all kwargs?
            _plot_time_evolution(ax[a], result, var, arange=arange, show_arange=kwargs['show_avg'],
                                 logy=kwargs['logy'], verbose=kwargs['verbose'], stats=kwargs['stats'])
            if a < len(vars)-1:
                ax[a].set_xticklabels([])

    # Ensure plot is exactly at simulation ends by default
    if kwargs['xmin'] is None:
        kwargs['xmin'] = max_this_invocation['tmin']
    if kwargs['xmax'] is None:
        kwargs['xmax'] = max_this_invocation['tmax']
    if kwargs['verbose']:
        print(f"Setting time range: {kwargs['xmin']} {kwargs['xmax']}")

    return fig

def eh_fluxes(results, kwargs):
    return _plot_time_evolutions(results, kwargs, vars=('mdot', 'phi_b', 'spinup', 'eff'))

def eh_fluxes_norms(results, kwargs):
    for res in results: res.tag = "Per-step"
    fig = _plot_time_evolutions(results, kwargs, vars=('mdot_per', 'phi_b_per', 'spinup_per', 'eff_per'))
    for res in results: res.tag = "Smoothed"
    _plot_time_evolutions(results, kwargs, vars=('smooth_mdot', 'smooth_phi_b', 'smooth_spinup', 'smooth_eff'), fig=fig)
    for res in results: res.tag = "Back half"
    return _plot_time_evolutions(results, kwargs, vars=('mdot', 'phi_b', 'spinup', 'eff'), fig=fig)

def eh_fluxes_smooth(results, kwargs):
    return _plot_time_evolutions(results, kwargs, vars=('smooth_mdot', 'smooth_phi_b', 'smooth_spinup', 'smooth_eff'))

def eh_fluxes_smoothmdot(results, kwargs):
    for result in results:
        result.avg_is_smooth = True
    return _plot_time_evolutions(results, kwargs, vars=('smooth_mdot', 'phi_b', 'spinup', 'eff'))

def eh_fluxes_old(results, kwargs):
    return _plot_time_evolutions(results, kwargs, vars=('mdot', 'phi_b', 'ldot', 'eff'))

def eh_fluxes_old_jet50(results, kwargs):
    return _plot_time_evolutions(results, kwargs, vars=('mdot', 'phi_b', 'ldot', 'eff_jet50'))

def eh_fluxes_raw(results, kwargs):
    return _plot_time_evolutions(results, kwargs, vars=('Mdot', 'Phi_b', 'Ldot', 'Edot'))

def eff_versions(results, kwargs):
    return _plot_time_evolutions(results, kwargs, vars=('eff_55', 'eff_5EH', 'eff_EHEH')) #, 'eff_jet50'))

def spinup(results, kwargs):
    return _plot_time_evolutions(results, kwargs, vars=('spinup',))

def jet_efficiency(results, kwargs):
    return _plot_time_evolutions(results, kwargs, vars=('spinup',))

def mdot_versions(results, kwargs):
    return _plot_time_evolutions(results, kwargs, vars=('Mdot_EH', 'Mdot_5', 'diag/Mdot_Flux', 'diag/neg_InnerX1RHO'))

def edot_versions(results, kwargs):
    return _plot_time_evolutions(results, kwargs, vars=('Edot_EH', 'Edot_5', 'diag/Edot_Flux', 'diag/neg_InnerX1T00'))


def eh_phi_versions(results, kwargs):
    return _plot_time_evolutions(results, kwargs, vars=('phi_b', '2x_phi_b_lower', '2x_phi_b_upper', 'phi_b_hemispheres'))

def edot_comparisons(results, kwargs):
    vars = ('smooth_Edot_EH', 'smooth_Edot_5')
    xsize = float(kwargs['fig_x']) if kwargs['fig_x'] is not None else 10
    ysize = float(kwargs['fig_y']) if kwargs['fig_y'] is not None else 10/4*len(results)
    fig, _ = plt.subplots(len(results), 1, figsize=(xsize, ysize))
    ax = fig.get_axes()
    for var in vars:
        arange = kwargs['arange'] if 'arange' in kwargs else None
        for a,result in enumerate(results):
            _plot_time_evolution(ax[a], result, var, per=kwargs['per'], arange=arange,
                                show_arange=kwargs['show_avg'], ymax=kwargs['ymax_eff'],
                                label=var, verbose=kwargs['verbose'])
            ax[a].set_ylabel(result.tag)

    if kwargs['fig_wspace'] is None:
        kwargs['fig_wspace'] = 0.4
    return fig

# TODO time versions to check whether a diag/analysis contains all timesteps


def _point_per_run(axis, results, var, to_plot, plot_vs, window=None, arange=-1000, selector=None, tag="",
                  print_time=False, print_only_time=False, model_shared_portion=0, no_print_flux=False, plotrc={},
                  **kwargs):
    if plot_vs == 'spin':
        get_xval = lambda model: model['a']
        get_modelname = lambda model: model.tag
    elif plot_vs == 'res':
        get_xval = lambda model: model['nx3']
        get_modelname = lambda model: model.tag

    # Dictionaries by "model" of lists by spin
    model_xvals = {}
    model_yvals = {}
    model_stds = {}
    model_times = {}
    title = ""
    # Run through the files and suck up everything, sorting by "model" not including spin
    for result in results:
        # If this thing is even readable...
        avg_slice = result.get_time_slice(*arange)
        if len(result['t'][avg_slice]) == 0:
            print("Skipping {}: no data fround for range {}".format(result.tag, arange))
            continue


        model = get_modelname(result)
        xval = get_xval(result)

        model = " ".join(model.split(" ")[model_shared_portion:])
        title = " ".join(model.split(" ")[:model_shared_portion])

        if selector is not None and not selector(model):
            continue

        if model not in model_xvals:
            model_xvals[model] = []
            model_yvals[model] = []
            model_stds[model] = []
            # Record times to print, to nearest 1k
            model_times[model] = (round(result['t'][avg_slice][0]/1000)*1000,
                                  round(result['t'][avg_slice][-1]/1000)*1000)
        model_xvals[model].append(xval)

        if to_plot in ('avg', 'avg_std'):
            val = np.mean(result['t/'+var][avg_slice])
            if to_plot == 'avg_std':
                model_stds[model].append(np.std(result['t/'+var][avg_slice]))
        elif to_plot == 'std':
            val = np.std(result['t/'+var][avg_slice])
        elif to_plot == 'std_rel':
            val = np.std(result['t/'+var][avg_slice]) / np.mean(result['t/'+var][avg_slice])
        model_yvals[model].append(val)
    
    # Then plot each model
    for model in model_xvals.keys():
        if print_only_time:
            mname = r"{}-{} $t_g$".format(*model_times[model])
        else:
            mname = tag + model + ("", r" ({}-{} $t_g$)".format(*model_times[model]))[print_time]
        if no_print_flux:
            mname = mname.replace("MAD","").replace("SANE","").replace("  "," ")
        if to_plot == 'avg_std':
            # Sort all arrrays by x value to avoid weird back and forth lines
            xvals, yvals, ystd = zip(*sorted(zip(model_xvals[model], model_yvals[model], model_stds[model]), key=lambda x: x[0]))
            axis.errorbar(xvals, yvals, yerr=ystd, fmt='.--', capsize=5, label=mname, **plotrc)
        else:
            xvals, yvals = zip(*sorted(zip(model_xvals[model], model_yvals[model]), key=lambda x: x[0]))
            axis.plot(xvals, yvals, '.--', label=mname, **plotrc)

    axis.set_title(title)
    axis.grid(True)

    if plot_vs == 'spin':
        axis.set_xlim(-1,1)
        axis.set_xlabel(r"Spin $a_*$")
    elif plot_vs == 'res':
        axis.set_xlabel(r"Radial resolution")
        # TODO 2^x, log?

    if window is not None:
        axis.set_xlim(window[:2])
        axis.set_ylim(window[2:])

    if to_plot in ('avg', 'avg_std'):
        axis.set_ylabel(r"$\langle" + pyharm.pretty(var, segment=True) + r"\rangle$", rotation=0, ha='right')
    elif to_plot == 'std':
        axis.set_ylabel(r"$\sigma \left(" + pyharm.pretty(var, segment=True) + r"\right)$", rotation=0, ha='right')
    elif to_plot == 'std_rel':
        axis.set_ylabel(r"$\frac{\sigma \left(" + pyharm.pretty(var, segment=True) + r"\right)}{\langle" + pyharm.pretty(var, segment=True) + r"\rangle}$", rotation=0, ha='right')


# Ready-made names: figsize, save name, etc. TODO handle kwargs not passed on to line plot
def std_vs_spin(results, kwargs, plotrc={}):
    fig, ax = plt.subplots(1,1)
    _point_per_run(ax, results, kwargs['varlist'][0], 'std', 'spin', plotrc=plotrc, **kwargs)
    return fig
def avg_vs_spin(results, kwargs, plotrc={}):
    fig, ax = plt.subplots(1,1)
    _point_per_run(ax, results, kwargs['varlist'][0], 'avg', 'spin', plotrc=plotrc, **kwargs)
    return fig
def avg_std_vs_spin(results, kwargs, plotrc={}):
    fig, ax = plt.subplots(1,1)
    _point_per_run(ax, results, kwargs['varlist'][0], 'avg_std', 'spin', plotrc=plotrc, **kwargs)
    return fig

def res_study_std(results, kwargs, plotrc={}):
    fig, ax = plt.subplots(1,1)
    _point_per_run(ax, results, kwargs['varlist'][0], 'std', 'res', plotrc=plotrc, **kwargs)
    return fig
def res_study_avg(results, kwargs, plotrc={}):
    fig, ax = plt.subplots(1,1)
    _point_per_run(ax, results, kwargs['varlist'][0], 'avg', 'res', plotrc=plotrc, **kwargs)
    return fig
def res_study_avg_std(results, kwargs, plotrc={}):
    fig, ax = plt.subplots(1,1)
    _point_per_run(ax, results, kwargs['varlist'][0], 'avg_std', 'res', plotrc=plotrc, **kwargs)
    return fig
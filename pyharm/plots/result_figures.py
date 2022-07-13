
import numpy as np
import matplotlib.pyplot as plt

import pyharm
from pyharm.defs import Loci

def _savefig(fig, tag, kwargs):
    # Any other closing/format stuff. PDF format, etc, etc
    for ax in fig.get_axes():
        ax.set_xlim(kwargs['xmin'],  kwargs['xmax'])
        ax.set_ylim(kwargs['ymin'],  kwargs['ymax'])
    plt.subplots_adjust(wspace=0.4)
    # Underscore-separate things
    if kwargs['tag'] != "":
        kwargs['tag'] += "_"
    # User tag, then fn, then extension
    if kwargs['pdf']:
        plt.savefig(kwargs['tag']+tag+".pdf", dpi=200)
    else:
        plt.savefig(kwargs['tag']+tag+".png", dpi=200)
    plt.close(fig)

def _get_t_slice(result, arange):
    """Returns a time slice corresponding to the tuple or number 'arange'
    (optionally negative-indexed from sim end)
    """
    # TODO BOUNDS CORRECTLY
    if isinstance(arange, slice) or isinstance(arange, tuple) or isinstance(arange, list):
        try:
            return result.get_time_slice(arange[0], arange[1])
        except KeyError:
            return None
    elif arange is not None:
        # Min only, negative offset from end accepted
        try:
            return result.get_time_slice(arange)
        except KeyError:
            return None
    else:
        return True, slice(None)

def _get_r_slice(result, rrange):
    """Get a slice of radial zones matching the plot window.
    Plotting only the necessary radial range makes auto-scaling of the y-axis work
    """
    return slice(max(pyharm.util.i_of(result['r'], rrange[0]), 0), pyharm.util.i_of(result['r'], rrange[1]))

def _remove_spin(tag):
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
        return folder


def initial_conditions(results, kwargs, overplot=False): # TODO radial_averages_at
    """
    """
    if kwargs['varlist'] is None:
        vars=('rho', 'Pg', 'b', 'bsq', 'Ptot', 'u^3', 'sigma_post', 'inv_beta_post')

def radial_profile(ax, result, var, arange=-1000, window=(1,50), disk=True, plot_std=False, plot_eh=False, print_time=False, selector=None, tag="", **kwargs):

    if selector is not None and not selector(model):
        return

    # Get the times to average
    avg_slice = _get_t_slice(result, arange)
    times = (round(result['t'][avg_slice][0]/1000)*1000,
             round(result['t'][avg_slice][-1]/1000)*1000)
    # Get just the relevant radial slice so y-limits get set properly
    r_slice = _get_r_slice(result, window)

    if disk:
        tyvals = result['rt/{}_disk'.format(var)][avg_slice, r_slice]
    else:
        try:
            tyvals = result['rt/{}_all'.format(var)][avg_slice, r_slice]
        except (OSError,IOError):
            tyvals = result['rt/{}_disk'.format(var)][avg_slice, r_slice] + result['rt/{}_notdisk'.format(var)][avg_slice, r_slice]

    yvals = np.mean(tyvals, axis=0)
    p = ax.loglog(result['r'][r_slice], yvals, label=tag+result.tag+("",r"({}-{} $t_g$)".format(*times))[print_time], **kwargs)
    if plot_std:
        yerrs = np.std(tyvals, axis=0)
        ax.fill_between(result['r'][r_slice], yvals-yerrs, yvals+yerrs, alpha=0.5, color=p[0].get_color())

    if plot_eh:
        ax.axvline(2.0, color='k')
    else:
        ax.set_xlim(2.0, None)

    ax.set_xlabel(r"Radius [$r_g$]")
    ax.set_ylabel(pyharm.pretty(var), rotation=0, ha='right')
    ax.legend()
    ax.grid(True)

def point_per_run(axis, results, var, to_plot, plot_vs, window=None, arange=-1000, selector=None, tag="", **kwargs):
    if plot_vs == 'spin':
        get_xval = lambda tag: float(tag.split(" ")[-1].lstrip("A"))
        get_modelname = _remove_spin
    elif plot_vs == 'res':
        get_xval = lambda tag: int(tag.split(" ")[-1].split("X")[0].split("x")[0])
        get_modelname = lambda tag: " ".join(tag.split(" ")[:-1])

    # Dictionaries by "model" of lists by spin
    model_xvals = {}
    model_yvals = {}
    model_stds = {}
    model_times = {}
    # Run through the files and suck up everything, sorting by "model" not including spin
    for result in results.values():
        # If this thing is even readable...
        avg_slice = _get_t_slice(result, arange)
        if avg_slice is None:
            print("Skipping {}: no data fround for range {}".format(result.tag, arange))
            continue


        model = get_modelname(result.tag)
        xval = get_xval(result.tag)

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
        if to_plot == 'avg_std':
            # Sort all arrrays by x value to avoid weird back and forth lines
            xvals, yvals, ystd = zip(*sorted(zip(model_xvals[model], model_yvals[model], model_stds[model]), key=lambda x: x[0]))
            axis.errorbar(xvals, yvals, yerr=ystd, fmt='.--', capsize=5, label=model + " ({} to {})".format(*model_times[model]), **kwargs)
        else:
            xvals, yvals = zip(*sorted(zip(model_xvals[model], model_yvals[model]), key=lambda x: x[0]))
            axis.plot(xvals, yvals, '.--', label=tag+model+r" ({}-{} $t_g$)".format(*model_times[model]), **kwargs)


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

    axis.legend()

# Ready-made names: figsize, save name, etc. TODO handle kwargs not passed on to line plot
def std_vs_spin(ax, results, kwargs):
    point_per_run(ax, results, kwargs['varlist'][0], 'std', 'spin', **kwargs)
def avg_vs_spin(results, kwargs):
    point_per_run(ax, results, kwargs['varlist'][0], 'avg', 'spin', **kwargs)
def avg_std_vs_spin(results, kwargs):
    point_per_run(ax, results, kwargs['varlist'][0], 'avg_std', 'spin', **kwargs)

def res_study_std(results, kwargs):
    point_per_run(ax, results, kwargs['varlist'][0], 'std', 'res', **kwargs)
def res_study_avg(results, kwargs):
    point_per_run(ax, results, kwargs['varlist'][0], 'avg', 'res', **kwargs)
def res_study_avg_std(results, kwargs):
    point_per_run(ax, results, kwargs['varlist'][0], 'avg_std', 'res', **kwargs)

# TODO dump. Radial stuff
def default_radial_averages(results, kwargs):
    if kwargs['vars'] is None:
        vars = ('rho', 'Pg', 'b', 'bsq', 'Ptot', 'u^3', 'sigma_post', 'inv_beta_post')
    else:
        vars = kwargs['vars']

    for result in results.values():
        # Radial profiles of variables
        nx = min(len(vars), 4)
        ny = (len(vars)-1)//4+1
        fig, _ = plt.subplots(ny, nx, figsize=(4*nx,4*ny))
        ax = fig.get_axes()

        window = plot_radial_averages(ax, results, kwargs, default_r=50)

        if window:
            _savefig(fig, "radial_averages_by_window_"+result.tag, kwargs)
        else:
            _savefig(fig, "radial_averages_"+result.tag, kwargs)
        

def radial_fluxes(results, kwargs):
    for result in results.values():
        fig, _ = plt.subplots(1,3, figsize=(14,4))
        ax = fig.get_axes()
        window = plot_radial_averages(ax, results, kwargs, default_r=20)
        if window:
            _savefig(fig, "radial_fluxes_by_window_"+result.tag, kwargs)
        else:
            _savefig(fig, "radial_fluxes_"+result.tag, kwargs)


def disk_momentum(results, kwargs):
    kwargs['vars'] = "u_3"
    return radial_averages(results, kwargs)

def plot_eh_fluxes(ax, result, per=False):
    if per:
        tag = '_per'
    else:
        tag = ''
    for a,var in enumerate(('mdot', 'phi_b'+tag, 'abs_ldot'+tag, 'eff'+tag)):
        ax[a].plot(result['t'], result['t/{}'.format(var)], label=result.tag)
        ax[a].set_ylabel(pyharm.pretty(var), rotation=0, ha='right')
        ax[a].grid(True)

def plot_eh_phi_versions(ax, result):
    for a,var in enumerate(('phi_b', 'phi_b_upper', 'phi_b_lower')):
        ax[a].plot(result['t'], result['t/{}'.format(var)], label=result.tag)
        ax[a].set_ylabel(pyharm.pretty(var), rotation=0, ha='right')
        ax[a].grid(True)
    # Additionally plot
    ax[0].plot(result['t'], np.abs(result['t/phi_b_upper'])+np.abs(result['t/phi_b_lower']), label=result.tag+" hemispheres")

def eh_fluxes(results, kwargs):
    for result in results.values():
        # Event horizon fluxes
        fig, _ = plt.subplots(4,1, figsize=(7,7))
        axes = fig.get_axes()
        plot_eh_fluxes(axes, result)
        plt.subplots_adjust(wspace=0.4)
        _savefig(fig, "eh_fluxes_"+result.tag, kwargs)

def eh_fluxes_per(results, kwargs):
    for result in results.values():
        print(result.fname)
        # Event horizon fluxes
        fig, _ = plt.subplots(4,1, figsize=(7,7))
        axes = fig.get_axes()
        plot_eh_fluxes(axes, result, per=True)
        plt.subplots_adjust(wspace=0.4)
        _savefig(fig, "eh_fluxes_per_"+result.tag, kwargs)

def eh_phi_versions(results, kwargs):
    for result in results.values():
        # Event horizon fluxes
        fig, _ = plt.subplots(3,1, figsize=(7,7))
        axes = fig.get_axes()
        plot_eh_phi_versions(axes, result)
        plt.subplots_adjust(wspace=0.4)
        _savefig(fig, "eh_phi_versions_"+result.tag, kwargs)

def overplot_eh_phi_versions(results, kwargs):
    for result in results.values():
        # Event horizon fluxes
        fig, _ = plt.subplots(3,1, figsize=(7,7))
        axes = fig.get_axes()
        plot_eh_phi_versions(axes, result)
        plt.subplots_adjust(wspace=0.4)
        _savefig(fig, "eh_phi_versions_"+result.tag, kwargs)

def overplot_eh_fluxes(results, kwargs):
    fig, _ = plt.subplots(4,1, figsize=(7,7))
    ax = fig.get_axes()
    for result in results.values():
        plot_eh_fluxes(ax, result)

    ax[0].legend()
    plt.subplots_adjust(wspace=0.4)
    _savefig(fig, "eh_fluxes_compare", kwargs)
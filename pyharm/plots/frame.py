import os, sys
import click

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .. import io
from ..fluid_dump import FluidDump

from . import figures
from .plot_dumps import *
from .pretty import pretty


"""Generate one frame of a movie.  Currently pretty useless outside `movie.py`,
present in pyharm so as to be importable by different MPI processes and across movie generation "schemes," if it comes to that.
"""

def frame(fname, diag, kwargs):
    tstart, tend = kwargs['tstart'], kwargs['tend']
    movie_type = kwargs['movie_type']
    tdump = io.get_dump_time(fname)
    frame_name = os.path.join("frames_"+movie_type, "frame_t%08d.png" % int(tdump))
    if (tstart is not None and tdump < tstart) or \
        (tend is not None and tdump > tend) or \
        ('resume' in kwargs and kwargs['resume'] and os.path.exists(frame_name)):
        return

    print("Imaging t={}".format(int(tdump)), file=sys.stderr)

    # Load ghosts?  Then strip the option out of the movie name
    ghost_zones = False
    if "_ghost" in movie_type:
        ghost_zones = True
        movie_type = movie_type.replace("_ghost","")

    # This attaches the file and creates a grid
    dump = FluidDump(fname, ghost_zones=ghost_zones)

    # Set some plot options
    plotrc = {}
    # Copy in the equivalent options, casting them to what below code expects
    for key in ('vmin', 'vmax', 'shading', 'native', 'cmap', 'at', 'bh', 'nlines'):
        if key in kwargs:
            plotrc[key] = kwargs[key]
            if key in ('vmin', 'vmax'):
                # Should be floats or none
                if plotrc[key] is not None:
                    plotrc[key] = float(plotrc[key])
            if key in ('at', 'nlines'):
                # Should be ints
                plotrc[key] = int(plotrc[key])
            if key in ('native', 'bh'):
                # Should be bools
                plotrc[key] = bool(plotrc[key])

    # Choose a domain size 
    if kwargs['size'] is not None:
        sz = float(kwargs['size'])
    else:
        if 'r_out' not in dump.params:
            # Exotic. Try plotting the whole domain
            sz = None
        else:
            # Mediocre heuristic for "enough" of domain.
            # Users should specify
            if dump['r_out'] >= 500:
                sz = dump['r_out']/10
            elif dump['r_out'] >= 100:
                sz = dump['r_out']/3
            else:
                sz = dump['r_out']

    # Choose a centered window
    # TODO 'half' and similar args for non-centered windows
    if sz is not None:
        plotrc['window'] = (-sz, sz, -sz, sz)
    else:
        plotrc['window'] = None

    #  _array plots override a bunch of things
    # Handle and strip
    if "_array" in movie_type:
        plotrc['native'] = True
        plotrc['window'] = None # Let plotter choose based on grid
        plotrc['shading'] = 'flat'
        plotrc['half_cut'] = True
        movie_type = movie_type.replace("_array","")

    # Options to place
    if "_cross" in movie_type:
        movie_type = movie_type.replace("_cross","")
        plotrc['at'] = dump['n3']//2
    if "_quarter" in movie_type:
        movie_type = movie_type.replace("_quarter","")
        plotrc['at'] = dump['n3']//4

    fig = plt.figure(figsize=(kwargs['fig_x'], kwargs['fig_y']))

    if movie_type in figures.__dict__:
        # Named movie frame figures in figures.py
        fig = figures.__dict__[movie_type](fig, dump, diag, plotrc)

    else:
        # Try to make a simple movie of just the stated variable

        # Strip off the usual annotations if we want something pretty
        no_margin = False
        if "_simple" in movie_type:
            no_margin = True
            plotrc.update({'xlabel': False, 'ylabel': False,
                           'xticks': [], 'yticks': [],
                           'cbar': False, 'frame': False})
            movie_type = movie_type.replace("_simple","")

        # Various options 
        if "_poloidal" in movie_type or "_2d" in movie_type:
            ax = plt.subplot(1, 1, 1)
            var = movie_type.replace("_poloidal","")
            plot_xz(ax, dump, var, **plotrc)
        elif "_toroidal" in movie_type:
            ax = plt.subplot(1, 1, 1)
            var = movie_type.replace("_toroidal","")
            plot_xy(ax, dump, var, **plotrc)
        elif "_1d" in movie_type:
            ax = plt.subplot(1, 1, 1)
            var = movie_type.replace("_1d","")
            sec = dump[:, 0, 0]
            ax.plot(sec['r'], sec[var]) # TODO some kind of radial_plot back in plot_dumps?
            ax.set_ylim((plotrc['vmin'], plotrc['vmax']))
            # TODO multiple variables w/user title?
            ax.set_title(pretty(var))
        else:
            ax_slc = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
            ax = ax_slc[0]
            var = movie_type
            #print("Plotting slices. plotrc: ", plotrc)
            plot_slices(ax_slc[0], ax_slc[1], dump, var, **plotrc) # We'll plot the field ourselves

        if no_margin or "jsq" in movie_type:
            fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)
        else:
            fig.subplots_adjust(left=0.03, right=0.97)

        if 'overlay_field' in kwargs and kwargs['overlay_field'] and not plotrc['native']:
            nlines = plotrc['nlines'] if 'nlines' in plotrc else 20
            overlay_field(ax, dump, nlines=nlines)
        # TODO contours

    # If the figure code didn't set the title
    # I cannot be bothered to flag this for myself
    if fig._suptitle is None or fig._suptitle.get_text() == "":
        if "divB" in movie_type:
            # Special title for diagnostic divB
            fig.suptitle(r"Max $\nabla \cdot B$ = {}".format(np.max(np.abs(dump['divB']))))
        else:
            # Title by time, otherwise number
            fig.suptitle("t = {}".format(int(tdump)))

    # Save by name, clean up
    plt.savefig(frame_name, dpi=kwargs['fig_dpi'])
    plt.close()
    del dump

    return fname
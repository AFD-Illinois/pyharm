__license__ = """
 File: frame.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020-2023, Ben Prather and AFD Group at UIUC
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

import os
import sys

import glob
import numpy as np

import matplotlib.pyplot as plt

from .. import io
from ..fluid_state import FluidState

from . import figures
from .plot_dumps import *
from .overlays import *
from .pretty import pretty

__doc__ = \
"""Generate one frame of a movie.  Currently pretty useless outside `pyharm-movie` script,
included in pyharm so as to be imported there easily.

The code in `figures` would be a better place to start in writing your own additional movies/plots.
"""

def do_plot(fig, dump, diag, movie_type, plotrc):
        # PLOT
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

            if "log_" in movie_type:
                movie_type = movie_type.replace("log_","")
                plotrc['log'] = True

            # Various options 
            if "_poloidal" in movie_type or "_2d" in movie_type:
                ax = plt.subplot(1, 1, 1)
                movie_type = movie_type.replace("_poloidal","")
                var = movie_type
                if "divB" in var:
                    var = dump[var]
                plot_xz(ax, dump, var, **plotrc)
            elif "_toroidal" in movie_type:
                ax = plt.subplot(1, 1, 1)
                movie_type = movie_type.replace("_toroidal","")
                var = movie_type
                if "divB" in var:
                    var = dump[var]
                plot_xy(ax, dump, var, **plotrc)
            elif "_av1d" in movie_type:
                ax = plt.subplot(1, 1, 1)
                movie_type = movie_type.replace("_av1d","")
                var = movie_type
                vardata = np.mean(dump[var], axis=(1,2))
                ax.plot(dump['r1d'], vardata) # TODO some kind of radial_plot back in plot_dumps?

                ax.set_ylim((plotrc['vmin'], plotrc['vmax']))
                if plotrc['log']:
                    ax.set_yscale('log')
                ax.set_xlim((plotrc['window'][0], plotrc['window'][1]))
                if plotrc['log_r']:
                    ax.set_xscale('log')
                # TODO multiple variables w/user title?
                ax.grid(True, axis='both')
                ax.set_title(pretty(var))
            elif "_1d" in movie_type:
                ax = plt.subplot(1, 1, 1)
                movie_type = movie_type.replace("_1d","")
                var = movie_type
                sec = dump[:, 0, 0]
                ax.plot(sec['r1d'], np.squeeze(sec[var])) # TODO some kind of radial_plot back in plot_dumps?

                ax.set_ylim((plotrc['vmin'], plotrc['vmax']))
                if plotrc['log']:
                    ax.set_yscale('log')
                if plotrc['window'] is not None:
                    ax.set_xlim((plotrc['window'][0], plotrc['window'][1]))
                if plotrc['log_r']:
                    ax.set_xscale('log')
                ax.grid(True, axis='both')
                ax.set_title(pretty(var))
            else:
                ax_slc = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
                ax = ax_slc[0]
                var = movie_type
                #print("Plotting slices. plotrc: ", plotrc)
                if "divB" in var:
                    var = dump[var]
                plot_slices(ax_slc[0], ax_slc[1], dump, var, **plotrc) # We'll plot the field ourselves

            if no_margin:
                fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)
            else:
                adjustrc = {}
                for key in ('left', 'right', 'top', 'bottom', 'wspace', 'hspace'):
                    if key in plotrc and plotrc[key] is not None:
                        adjustrc[key] = plotrc[key]
                fig.subplots_adjust(**adjustrc)

def frame(fname, diag, kwargs):
    # If we're outside the timeframe we don't need to make *anything*
    tstart, tend = kwargs['tstart'], kwargs['tend']
    tdump = io.get_dump_time(fname)
    if tdump is None:
        # TODO yell about not knowing dump times
        return
    if (tstart is not None and tdump < float(tstart)) or \
        (tend is not None and tdump > float(tend)):
        return

    # Check through movies for which we've run/need to run,
    # and which will include ghosts
    movie_types = []
    ghost_zones = False
    for movie_type in kwargs['movie_types'].split(","):
        frame_folder = os.path.join(os.getcwd().replace(kwargs['base_path'], kwargs['out_path']), "frames_"+movie_type)
        if 'numeric_fnames' in kwargs and kwargs['numeric_fnames']:
            frame_name = os.path.join(frame_folder, "frame_"+fname.split('.')[-2]+".png")
        elif 'accurate_fnames' in kwargs and kwargs['accurate_fnames']:
            time_formatted = ("%.2f"%tdump).rjust(kwargs['time_digits'],'0')
            frame_name = os.path.join(frame_folder, "frame_t"+time_formatted+".png")
        else:
            time_formatted = ("%d"%int(tdump)).rjust(kwargs['time_digits'],'0')
            frame_name = os.path.join(frame_folder, "frame_t"+time_formatted+".png")

        if 'resume' in kwargs and kwargs['resume'] and os.path.exists(frame_name):
            continue

        # Load ghosts?  Set a flag and strip the option from the name
        if "_ghost" in movie_type or kwargs['ghost_zones']:
            ghost_zones = True
            movie_type = movie_type.replace("_ghost","")

        # Then add the stripped name to the list
        movie_types.append(movie_type)

    # If we don't have any frames to make, return
    if len(movie_types) == 0:
        return

    print("Imaging t={}".format(int(tdump)), file=sys.stderr)

    # This just attaches the file and creates a grid.  We do need to specify
    # if any movie will need ghosts, for the index math
    dump = FluidState(fname, ghost_zones=ghost_zones, use_grid_cache=(not kwargs['no_grid_cache']), multizone=kwargs['multizone'])

    for movie_type in movie_types:
        # Set plotting options we'll pass on to figure-specific code
        plotrc = {}
        # Plotting options are copied from kwargs and share the same names
        for key in ('vmin', 'vmax', 'xmin', 'xmax', 'ymin', 'ymax',
                    'left', 'right', 'top', 'bottom', 'wspace', 'hspace'):
            # Should be floats or none
            try:
                plotrc[key] = float(kwargs[key])
            except TypeError:
                # Make everything else None
                plotrc[key] = None
        for key in ('at', 'nlines'):
            # Should be ints
            plotrc[key] = int(kwargs[key])
        for key in ('native', 'bh', 'no_title', 'average', 'sum', 'log', 'log_r'):
            # Should be bools
            plotrc[key] = bool(kwargs[key])
        for key in ('shading', 'cmap'):
            plotrc[key] = kwargs[key] #lower()?

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
                # TODO at least r_in to e.g. 10*r_in or something
                if dump['r_out'] >= 60:
                    sz = 60
                else:
                    sz = dump['r_out']

        # Choose a centered window
        # TODO 'half' and similar args for non-centered windows
        user_window = False
        if kwargs['xmin'] is not None:
            plotrc['window'] = (plotrc['xmin'], plotrc['xmax'], plotrc['ymin'], plotrc['ymax'])
            user_window = True
        elif sz is not None:
            if "1d" in movie_type:
                plotrc['window'] = (1.0, sz)
            else:
                plotrc['window'] = (-sz, sz, -sz, sz)
        else:
            plotrc['window'] = None

        # If our plot would be entirely outside the active window
        # TODO account for log_r
        # if user_window:
        #     if dump['r_in'] > plotrc['xmax'] and dump['r_in'] > plotrc['ymax'] and \
        #         -dump['r_in'] < plotrc['xmin'] and -dump['r_in'] < plotrc['ymin']:
        #         return
        #     if 'r_in_active' in dump.params:
        #         if dump['r_in_active'] > plotrc['xmax'] and dump['r_in_active'] > plotrc['ymax'] and \
        #             -dump['r_in_active'] < plotrc['xmin'] and -dump['r_in_active'] < plotrc['ymin']:
        #             return

        #  _array plots override a bunch of things
        # Handle and strip
        plotrc['native'] = False
        if "_array" in movie_type:
            plotrc['native'] = True
            if not user_window:
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
        if "_poloidal" in movie_type:
            pass

        plotrc['overlay_field'] = \
            'overlay_field' in kwargs and kwargs['overlay_field'] and not plotrc['native']

        fig = plt.figure(figsize=(kwargs['fig_x'], kwargs['fig_y']))
        
        # Plot the dump we were assigned
        do_plot(fig, dump, diag, movie_type, plotrc)

        if kwargs['multizone']:
            # plot outlines of the current run *above* the current run
            # use circles to avoid contour computation/ugliness
            for ax in fig.axes:
                if plotrc['native']:
                    ax.axvline(dump['startx1_active'], color='r')
                    ax.axvline(dump['stopx1_active'], color='r')
                else:
                    rin = np.log(dump['r_in_active']) if plotrc['log_r'] else dump['r_in_active']
                    rout = np.log(dump['r_out_active']) if plotrc['log_r'] else dump['r_out_active']
                    ax.add_artist(plt.Circle((0, 0), rin, facecolor=(0,0,0,0), edgecolor='r'))
                    ax.add_artist(plt.Circle((0, 0), rout, facecolor=(0,0,0,0), edgecolor='r'))

        # OVERLAYS
        ax = fig.axes[0]
        if plotrc['overlay_field']:
            nlines = plotrc['nlines'] if 'nlines' in plotrc else 20
            overlay_field(ax, dump, nlines=nlines, native=plotrc['native'], log_r=plotrc['log_r'])
        if 'overlay_grid' in kwargs and kwargs['overlay_grid']:
            overlay_grid(ax, dump.grid, kwargs['overlay_grid_spacing'], native=plotrc['native'], log_r=plotrc['log_r'])
        if 'overlay_blocks' in kwargs and kwargs['overlay_blocks']:
            overlay_blocks(ax, dump, native=plotrc['native'], log_r=plotrc['log_r'])
            if len(fig.axes) > 2: # colorbar is an axis
                overlay_blocks_xy(fig.axes[1], dump, native=plotrc['native'], log_r=plotrc['log_r'])


        # TODO options here, contours, etc.

        # TITLE
        # Always quash title when set. figures can set this too
        if plotrc['no_title']:
            fig.suptitle("")
        elif (fig._suptitle is None or fig._suptitle.get_text() == ""):
            # If the figure didn't set a title and we should...
            if "divB" in movie_type:
                # Special title for diagnostic divB
                # movie_type might be a version calculated in post e.g. divB_prims
                divb = dump[movie_type.replace("_poloidal","").replace("log_","")]
                divb_max = np.nanmax(divb)
                divb_argmax = np.nanargmax(divb)
                fig.suptitle(r"Max $\nabla \cdot B$ = {}".format(divb_max))
                print("divB max", divb_max, "at", np.unravel_index(divb_argmax, divb.shape))
            else:
                # Title by tdump, which is time if available, else dump number
                fig.suptitle("t = {}".format(int(tdump)))

        # Save by name, clean up
        plt.savefig(frame_name, dpi=kwargs['fig_dpi'])
        plt.close()

    del dump
    return len(movie_types)

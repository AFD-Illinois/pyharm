__license__ = """
 File: frame.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020-2022, AFD Group at UIUC
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

import numpy as np

import matplotlib.pyplot as plt

from .. import io
from ..fluid_dump import FluidDump

from . import figures
from .plot_dumps import *
from .overlays import *
from .pretty import pretty

__doc__ = \
"""Generate one frame of a movie.  Currently pretty useless outside `pyharm-movie` script,
included in pyharm so as to be imported there easily.

The code in `figures` would be a better place to start in writing your own additional movies/plots.
"""

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
        if 'accurate_fnames' in kwargs and kwargs['accurate_fnames']:
            time_formatted = ("%.2f"%tdump).rjust(kwargs['time_digits'],'0')
            frame_name = os.path.join(frame_folder, "frame_t"+time_formatted+".png")
        else:
            time_formatted = ("%d"%int(tdump)).rjust(kwargs['time_digits'],'0')
            frame_name = os.path.join(frame_folder, "frame_t"+time_formatted+".png")

        if 'resume' in kwargs and kwargs['resume'] and os.path.exists(frame_name):
            continue

        # Load ghosts?  Set a flag and strip the option from the name
        if "_ghost" in movie_type:
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
    dump = FluidDump(fname, ghost_zones=ghost_zones, grid_cache=(not kwargs['no_grid_cache']))

    for movie_type in movie_types:
        # Set some plot options
        plotrc = {}
        # Copy in the equivalent options, casting them to what below code expects
        for key in ('vmin', 'vmax', 'xmin', 'xmax', 'ymin', 'ymax', # float
                    'left', 'right', 'top', 'bottom', 'wspace', 'hspace', # float
                    'at', 'nlines', # int
                    'native', 'bh', 'no_title', 'average', 'sum', 'log', 'log_r', # bool
                    'shading', 'cmap'): # string
            if key in kwargs:
                plotrc[key] = kwargs[key]
                if key in ('vmin', 'vmax', 'xmin', 'xmax', 'ymin', 'ymax',
                            'left', 'right', 'top', 'bottom', 'wspace', 'hspace'):
                    # Should be floats or none
                    if plotrc[key] is not None and plotrc[key] != "None":
                        plotrc[key] = float(plotrc[key])
                    else:
                        plotrc[key] = None
                if key in ('at', 'nlines'):
                    # Should be ints
                    plotrc[key] = int(plotrc[key])
                if key in ('native', 'bh', 'no_title', 'average', 'sum', 'log', 'log_r'):
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
            plotrc['window'] = (-sz, sz, -sz, sz)
        else:
            plotrc['window'] = None

        #  _array plots override a bunch of things
        # Handle and strip
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

        fig = plt.figure(figsize=(kwargs['fig_x'], kwargs['fig_y']))

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

            # Set "rho" movies to have a consistent colorbar
            if "log_rho" in movie_type:
                if plotrc['vmin'] is None:
                    plotrc['vmin'] = -4
                if plotrc['vmax'] is None:
                    plotrc['vmax'] = 1.5

            # Various options 
            if "_poloidal" in movie_type or "_2d" in movie_type:
                ax = plt.subplot(1, 1, 1)
                var = movie_type.replace("_poloidal","")
                if "divB" in var:
                    var = dump[var]
                plot_xz(ax, dump, var, **plotrc)
            elif "_toroidal" in movie_type:
                ax = plt.subplot(1, 1, 1)
                var = movie_type.replace("_toroidal","")
                if "divB" in var:
                    var = dump[var]
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

        # OVERLAYS
        if 'overlay_field' in kwargs and kwargs['overlay_field'] and not ('native' in plotrc and plotrc['native']):
            nlines = plotrc['nlines'] if 'nlines' in plotrc else 20
            overlay_field(ax, dump, nlines=nlines)
        if 'overlay_grid' in kwargs and kwargs['overlay_grid']:
            overlay_grid(ax, dump.grid)
        # TODO contours

        # TITLE
        # Always quash title when set. figures can set this too
        if plotrc['no_title']:
            fig.suptitle("")
        elif (fig._suptitle is None or fig._suptitle.get_text() == ""):
            # If the figure didn't set a title and we should...
            if "divB" in movie_type:
                # Special title for diagnostic divB
                divb = dump[movie_type]
                divb_max = np.max(divb)
                divb_argmax = np.argmax(divb)
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

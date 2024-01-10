
import numpy as np
import matplotlib
import yt

from yt.visualization.api import PlotCallback
from unyt import unyt_array

class SphGridBoundaryCallback(PlotCallback):
    _type_name = "grids_spherical"
    
    def __init__(self, ds, color='k', linewidth=0.5):
        self.ds = ds
        self.color = color
        self.linewidth = linewidth

    def __call__(self, plot):
        xlim, ylim = plot._axes.get_xlim(), plot._axes.get_ylim()
        # this method's signature is required
        # this is where we perform potentially expensive operations
        block_bounds = self.ds.get_block_boundaries()
        
        # the plot argument exposes matplotlib objects:
        # - plot._axes is a matplotlib.axes.Axes object
        # - plot._figure is a matplotlib.figure.Figure object
        ax = plot._axes
        c = self.ds.transform
        for bb in block_bounds:
            line1 = np.array([[l,bb[2],0] for l in np.linspace(bb[0], bb[1])]).T
            line2 = np.array([[l,bb[3],0] for l in np.linspace(bb[0], bb[1])]).T
            line3 = np.array([[bb[0],l,0] for l in np.linspace(bb[2], bb[3])]).T
            line4 = np.array([[bb[1],l,0] for l in np.linspace(bb[2], bb[3])]).T
            ax.plot(c.cart_x(line1), c.cart_z(line1), color=self.color, linewidth=self.linewidth)
            ax.plot(c.cart_x(line2), c.cart_z(line2), color=self.color, linewidth=self.linewidth)
            ax.plot(c.cart_x(line3), c.cart_z(line3), color=self.color, linewidth=self.linewidth)
            ax.plot(c.cart_x(line4), c.cart_z(line4), color=self.color, linewidth=self.linewidth)

        plot._axes.set_xlim(xlim)
        plot._axes.set_ylim(ylim)

class AxisLabelsCallback(PlotCallback):
    _type_name = "grav_units"

    def __init__(self):
        pass

    def __call__(self, plot):
        ax = plot._axes
        xvar = ax.get_xlabel()[5:6]
        ax.set_xlabel(xvar+r" ($r_g$)")
        yvar = ax.get_ylabel()[5:6]
        ax.set_ylabel(yvar+r" ($r_g$)")

class EventHorizonCallback(PlotCallback):
    _type_name = "event_horizon"

    def __init__(self, ds, color='k'):
        self.ds = ds
        self.color = color

    def __call__(self, plot):
        # TODO ping ds for spin, spacetime?
        eh = matplotlib.patches.Circle((0,0), 1 + np.sqrt(1 - 0.9375**2), fill=True, color=self.color)
        plot._axes.add_artist(eh)
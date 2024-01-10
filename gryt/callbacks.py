
import numpy as np
import matplotlib
import yt

from yt.visualization.api import PlotCallback
from unyt import unyt_array

class SphGridBoundaryCallback(PlotCallback):
    _type_name = "grids_spherical"
    
    def __init__(self, ds, at=None, color='k', linewidth=0.5):
        # The plot object's ds is lies
        # Preserve the original to query the backing file
        self.ds = ds
        self.at = at
        self.color = color
        self.linewidth = linewidth

    def __call__(self, plot):
        # plot.xlim/ylim
        xlim, ylim = plot._axes.get_xlim(), plot._axes.get_ylim()
        block_bounds = self.ds.get_block_boundaries()
        
        at = self.at
        ax = plot._axes
        c = self.ds.transform
        if plot.frb.axis == 2:
            if at is None: # Assume phi=0. Probably not important
                at = 0.0
            for bb in block_bounds:
                line1 = np.array([[l,bb[2],at] for l in np.linspace(bb[0], bb[1])]).T
                line2 = np.array([[l,bb[3],at] for l in np.linspace(bb[0], bb[1])]).T
                line3 = np.array([[bb[0],l,at] for l in np.linspace(bb[2], bb[3])]).T
                line4 = np.array([[bb[1],l,at] for l in np.linspace(bb[2], bb[3])]).T
                ax.plot(c.cart_x(line1), c.cart_z(line1), color=self.color, linewidth=self.linewidth)
                ax.plot(c.cart_x(line2), c.cart_z(line2), color=self.color, linewidth=self.linewidth)
                ax.plot(c.cart_x(line3), c.cart_z(line3), color=self.color, linewidth=self.linewidth)
                ax.plot(c.cart_x(line4), c.cart_z(line4), color=self.color, linewidth=self.linewidth)
        else:
            if at is None: # Assume midplane
                at = (self.ds.domain_left_edge[1] + self.ds.domain_right_edge[1]) / 2
            for bb in block_bounds:
                line1 = np.array([[l,at,bb[4]] for l in np.linspace(bb[0], bb[1])]).T
                line2 = np.array([[l,at,bb[5]] for l in np.linspace(bb[0], bb[1])]).T
                line3 = np.array([[bb[0],at,l] for l in np.linspace(bb[4], bb[5])]).T
                line4 = np.array([[bb[1],at,l] for l in np.linspace(bb[4], bb[5])]).T
                ax.plot(c.cart_x(line1), c.cart_y(line1), color=self.color, linewidth=self.linewidth)
                ax.plot(c.cart_x(line2), c.cart_y(line2), color=self.color, linewidth=self.linewidth)
                ax.plot(c.cart_x(line3), c.cart_y(line3), color=self.color, linewidth=self.linewidth)
                ax.plot(c.cart_x(line4), c.cart_y(line4), color=self.color, linewidth=self.linewidth)

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
        # TODO this should be recorded in Params & therefore ds.parameters proper
        a = float(plot.ds.parameters['input']['coordinates']['a'])
        # TODO units probably
        eh = matplotlib.patches.Circle((0,0), 1 + np.sqrt(1 - a**2), fill=True, color=self.color)
        plot._axes.add_artist(eh)
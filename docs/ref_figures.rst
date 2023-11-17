.. _ref_figures:

Figures and Finished Plots
==========================
``pyharm`` includes a long list of different movie and plot types with which one can visualize a particular dump file.  The names of the following functions can all be passed to ``pyharm-movie`` to generate frames of that type, ranging from imaging just one dump to many entire runs.  In addition, if passed a key corresponding to a ``FluidState`` variable instead of a particular movie name (e.g. the examples in :ref:`keys`), it will just attempt to plot that variable.

Keys when plotting
------------------
In addition to the title or variable name, there are a few modifiers parsed specifically when requesting plots, which generally come *after* any relevant variable. The list of postfix options can be gleaned from :mod:`pyharm.plots.frame` but includes at least:

* `_array` for plots in native coordinates on the Cartesian logical grid
* `_poloidal` or `_2d`, `_toroidal` for plotting just one slice or plotting 2D simulation output
* `_1d` likewise for radial/1D simulation output
* `_ghost` for plotting ghost zones (works best with `_array`)
* `_simple` to turn off axis labels, ticks, and colorbars for a presentation or outreach movie, or if you just need the screen real-estate

There is one exception, which was written to emulate the `log_` unary operator for fluid dumps: `symlog_`.  Prefixing `symlog_` will plot using :func:`pyharm.plots.plot_utils.pcolormesh_symlog`, which can plot signed variables in a consistent two-tone log scale.  In addition, `log_` is reserved to be able to have a different function when plotting -- in the future it will just set the axis or colorbar when plotting, rather than taking the :math:`log_{10}` of the variable.

Figures
-------
Here is the full list of available figures.  Most figures should be compatible with most of the above modifiers, but it is not hard to make silly or incompatible combinations, and no effort is made to prevent this in ``pyharm``.

.. automodule:: pyharm.plots.figures
   :members:

Pretty Names
------------
``pyharm`` has some tools for translating to (or trying to guess) the "pretty" LaTeX name of a variable based on its name in ``pyharm``.  Generally this just involves calling :func:`pyharm.pretty(varname)`, with the rest happening in the background, but should it be necessary here's the full doc.

.. automodule:: pyharm.plots.pretty
   :members:
.. _ref_figures:

Figures and Finished Plots
==========================
``pyharm`` includes a long list of different movie and plot types with which one can visualize a particular dump file.

Keys when plotting
------------------
When running `pyharm-movie`, there are a few modifiers specific to requesting plots, which generally come *after* any relevant variable. The list of postfix options can be gleaned from :mod:`pyharm.plots.frame` but includes at least:
* `_array` for plots in native coordinates on the Cartesian logical grid
* `_poloidal` or `_2d`, `_toroidal` for restricting plotting to just one slice or plotting 2D simulation output
* `_1d` likewise for radial/1D simulation output
* `_ghost` for plotting ghost zones (works best with `_array`)
* `_simple` to turn off axis labels, ticks, and colorbars for a presentation or outreach movie, or if you just need the screen real-estate

There is one exception, which was written to emulate the `log_` unary operator for fluid dumps: `symlog_`.  Prefixing `symlog_` will plot using :func:`pyharm.plots.plot_utils.pcolormesh_symlog`

Figures
-------
Here is the full list/

.. automodule:: pyharm.plots.figures
   :members:

Pretty Names
------------
``pyharm`` has some tools for translating to (or trying to guess) the "pretty" LaTeX name of a variable based on its name in ``pyharm``.  Generally this just involves calling :func:`pyharm.pretty(varname)`, with the rest happening in the background, but should it be necessary here's the full doc.

.. automodule:: pyharm.plots.pretty
   :members:
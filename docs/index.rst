Welcome to pyharm's documentation!
==================================

This is the documentation for `pyharm <https://github.com/AFD-Illinois/pyharm>`_, a Python package for analyzing GRMHD results.  Specifically, ``pyharm`` is a library for reading, plotting, and analyzing the output of GRMHD codes implemented in the style of HARM `Gammie et. al. (2003) <https://doi.org/10.1086/374594>`_, written in Python.  It includes tools for working with the output from several GRMHD codes, notably the `iharm3d <https://github.com/AFD-Illinois/iharm3d>`_ and `KHARMA <https://github.com/AFD-Illinois/kharma>`_ HDF5 output formats.

``pyharm`` is centered around the :class:`pyharm.fluid_state.FluidState` object, which corresponds to a particular GRMHD output file, but allows accessing a bunch of additional derived properties as though they were members of the file, by calculating & caching them on the fly.  This is used extensively by the included scripts to allow plotting many different variables simply by specifying them on the
command line.

You might want to start with the documentation on the different :ref:`keys` that are supported directly by ``FluidState``, or on the available :ref:`ref_figures` and :ref:`ref_analyses` functions, which can provide a template for any plotting and reductions.  Anything added to these files can automatically be parallelized and controlled with the ``pyharm-movie`` and ``pyharm-analysis`` scripts.

This documentation sometimes lags development, feel free to submit bugs at the GitHub page when things don't work as described.

.. toctree::
   :maxdepth: 2

   installing
   keys
   ref_analyses
   ref_figures
   ref_ana_results
   ref_coords
   ref_defs
   ref_fluid_state
   ref_io
   ref_parameters
   ref_plots
   ref_units

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

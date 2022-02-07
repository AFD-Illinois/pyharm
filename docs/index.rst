Welcome to pyHARM's documentation!
==================================

This is the documentation for `pyHARM <https://github.com/AFD-Illinois/pyHARM>`_, a Python package for GRMHD.  It may be incomplete, as pyHARM is still very much in development.

Specifically, pyHARM is a library for dealing with the output of GRMHD codes implemented in the style of HARM `Gammie et. al. (2003) <https://doi.org/10.1086/374594>`_, written in Python.  It includes tools for working with the output from several GRMHD codes, notably the ``iharm3d`` HDF5 format (see :ref:`ref_dumps`).

Some example scripts for performing a set of analysis reductions, basic plotting operations, and geometry manipulation and recording can be found in the ``script/`` directory of the repository; they may be more up-to-date than the documentation.

.. toctree::
   :maxdepth: 2

   installing
   variables
   ref_coords
   ref_dumps
   ref_plots

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

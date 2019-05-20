Welcome to pyHARM's documentation!
==================================

This is the preliminary documentation for `pyHARM <https://github.com/AFD-Illinois/pyHARM>`_, a Python package for GRMHD.

Specifically, pyHARM is a re-implementation of the HARM algorithm from `Gammie et. al. (2003) <https://doi.org/10.1086/374594>`_, written in Python, optionally accelerated with OpenCL kernels written in `loopy <https://mathema.tician.de/software/loopy/>`_.  It also includes tools for working with the output from other GRMHD codes, specifically the HARM HDF5 format (see :ref:`ref_dumps`).

Some example scripts for performing a set of analysis reductions and basic plotting operations can be found in the ``script/`` directory of the repository; they may be more up-to-date than the documentation.

.. toctree::
   :maxdepth: 2
   :caption: Module Reference:

   ref_coords
   ref_dumps
   ref_plots

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

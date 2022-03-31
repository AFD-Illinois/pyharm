# pyharm
Python tools for HARM analysis.

`pyharm` is a library of Python functions and scripts for analyzing and plotting the output of General-Relativistic Magnetohydrodynamic (GRMHD) simulations.  It includes functions for obtaining a long list of different variables of interest (pressures, temperatures, stress-energy tensor, synchrotron emissivity) based on local fluid state.  It additionally includes reductions for performing surface and volume integrals of various types, sophisticated plotting tools, and MPI-accelerated scripts for quickly and efficiently producing movies and time-summed or -averaged reductions over large-scale simulations.

`pyharm` primarily supports simulations based on the HARM scheme (Gammie et al. 2003) -- [KHARMA](https://github.com/AFD-Illinois/kharma), [iharm3d](https://github.com/AFD-Illinois/iharm3d), and [ebhlight](https://github.com/AFD-Illinois/ebhlight).  It includes Python re-implementations of core parts of the scheme, useful for deriving everything about the simulation not directly present in output files.  It includes limited support for several other codes, either directly or after translation with [EHT-babel](https://github.com/AFD-Illinois/EHT-babel/).

The core of pyharm is the `FluidDump` object, which behaves similar to a Python dictionary of `numpy` arrays, but calculates its members on the fly by reading the original file and performing operations only as necessary ("lazy" evaluation).  `FluidDump` objects can be sliced similarly to `numpy` arrays, and subsequent file reads and calculations will be done over only the sliced portion.

Finally, as a consequence of needing to read GRMHD output, pyharm includes definitions of various coordinate systems in Kerr spacetime, as well as tools for dealing with a evenly spaced logically Cartesian grids under many different coordinate systems and transformations.  These might be independently useful, see `coordinates.py` & `grid.py` if you're looking for just coordinate tools.

## Installing:
The preferred installation method is to run simply:
```bash
$ pip3 install -e .
```
Thereafter pyharm should be importable from any Python prompt or script run in the same environment.  This command installs `pyharm` as "editable," so that changes to the source will be reflected immediately when next importing the package -- this is recommended, as it is expected that users will have to modify the source code eventually.  `pyharm` can also be installed as a user or system package, but at the cost of less easily modifying the source.

## Examples:
Perhaps the best reference for pyharm outside the source code is the documentation [here](https://pyharm.readthedocs.io/en/latest/). Or for the dev branch, [here](https://pyharm.readthedocs.io/en/dev/).  The documentation is light on examples, however.

One good starting point learning to use `pyharm` is to try out the existing tools in the `scripts/` directory.  These are quite capable wrappers for large swaths of `pyharm`'s functionality, and are added to the system `$PATH` variable upon installation to make them easy to call.  They support MPI parallelization for use at scale on big clusters (sample batch scripts for some Illinois-relevant clusters are included in `scripts/batch/`).

The two main scripts are `pyharm-movie` for producing plots and movies, and `pyharm-analysis` for performing a set of reductions over a full simulation's output.  If what you need to do involves making a movie or performing a reduction, it can often be implemented as a plugin to these existing scripts -- see the examples in `pyharm.plots.figures` for movies, and in `pyharm.ana.analyses` for reductions & computations.  New additions to those files will automatically become valid arugments to `pyharm-movie` and `pyharm-analysis`.

For more advanced usage, the `notebooks` directory has a sample Jupyter notebook playing around with some basic reductions & plots.

## Keys:
Several parts of `pyharm` try to parse strings to determine behavior, to specify a desired variable, plot, or result from the command line or quickly in notebooks.  These replace what might traditionally be member functions of e.g. the `FluidDump` object (think `dump.ucov()`) with what acts like a giant Python dictionary (`dump['ucov']`). Each "item" is computed at first access and cached thereafter.

Since it's possible to combine keys in arbitrary ways, there is no master list of valid keys, nor does it make sense to write e.g. `for key in dump` or `if "key" in dump`.  The most reliable way to determine whether something can be computed is to try it, and catch the `ValueError` (for unknown keys) or `IOError` (for known keys not present in a file) if it is not found.

That said, a good starter list, with references to more complete lists, can be found in the [documentation](https://pyharm.readthedocs.io/en/latest/keys.html).

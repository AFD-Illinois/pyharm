# pyharm
Python tools for HARM analysis.  Full documentation is [here](https://pyharm.readthedocs.io/en/latest/).

`pyharm` is a library of Python functions and scripts for analyzing and plotting the output of General-Relativistic Magnetohydrodynamic (GRMHD) simulations.  It includes functions for obtaining a long list of different variables of interest (pressures, temperatures, stress-energy tensor, synchrotron emissivity) based on local fluid state.  It additionally includes reductions for performing surface and volume integrals of various types, sophisticated plotting tools, and MPI-accelerated scripts for quickly and efficiently producing movies and time-summed or -averaged reductions over large-scale simulations.

`pyharm` primarily supports simulations based on the HARM scheme (Gammie et al. 2003) -- [KHARMA](https://github.com/AFD-Illinois/kharma), [iharm3d](https://github.com/AFD-Illinois/iharm3d), and [ebhlight](https://github.com/AFD-Illinois/ebhlight).  It includes Python re-implementations of core parts of the scheme, useful for deriving everything about the simulation not directly present in output files.  It includes limited support for several other codes, either directly or after translation with [EHT-babel](https://github.com/AFD-Illinois/EHT-babel/).

The core of pyharm is the `FluidState` object, which behaves similar to a Python dictionary of `numpy` arrays, but calculates its members on the fly by reading the original file and performing operations only as necessary ("lazy" evaluation).  `FluidState` objects can be sliced similarly to `numpy` arrays, and subsequent file reads and calculations will be done over only the sliced portion.  Between lazy evaluation and heavy parallelization through `multiprocessing` and `MPI`, `pyharm` is fast and scalable -- the default scripts are able to process TB of simulation output per minute.

As a consequence of needing to read GRMHD output, pyharm includes definitions of various coordinate systems in Kerr spacetime, as well as tools for dealing with a evenly spaced logically Cartesian grids under many different coordinate systems and transformations.  These might be independently useful, see `coordinates.py` & `grid.py` if you're looking for just coordinate tools.

## Installing:
The preferred installation method is to run simply:
```bash
$ pip3 install -e .
```
Thereafter pyharm should be importable from any Python prompt or script run in the same environment.  This command installs `pyharm` as "editable," so that changes to the source will be reflected immediately when next importing the package -- this is recommended, as it is expected that users will have to modify the source code eventually.  `pyharm` can also be installed as a user or system package, but at the cost of less easily modifying the source.

## Examples:
If you're skeptical of using a big library just to read HDF5 files, the `notebooks` directory has a sample Jupyter notebook playing around with some of the things that make `pyharm` cool & potentially useful to you.  The full developer reference is [here](https://pyharm.readthedocs.io/en/latest/). Or for the dev branch, [here](https://pyharm.readthedocs.io/en/dev/).

If you're more interested in ready-made tools, try calling `pyharm-movie` for producing plots and movies, and `pyharm-analysis` for performing reductions over a full simulation's output.  These and the other `pyharm` scripts are added to your `$PATH` upon installation, so they're always available.  There are quite a few different movies and analyses implemented, and new ones can be added easily by adding to `pyharm.plots.figures` to extend `pyharm-movie`, or `pyharm.ana.analyses` to extend `pyharm-analysis`.  New additions to those files will automatically become valid arugments to `pyharm-movie` and `pyharm-analysis`, show up as options in the help, etc.

## Keys:
Several parts of `pyharm` try to parse strings to determine behavior, to specify a desired variable, plot, or result from the command line or quickly in notebooks.  These replace what might traditionally be member functions of e.g. the `FluidState` object (think `dump.ucov()`) with what acts like a giant Python dictionary (`dump['ucov']`). Each "item" is computed at first access and cached thereafter.

Since it's possible to combine keys in arbitrary ways, there is no master list of valid keys, nor does it make sense to write e.g. `for key in dump` or `if "key" in dump`.  The most reliable way to determine whether something can be computed is to try it, and catch the `ValueError` (for unknown keys) or `IOError` (for known keys not present in a file) if it is not found.

That said, a good starter list, with references to more complete lists, can be found in the [documentation](https://pyharm.readthedocs.io/en/latest/keys.html).

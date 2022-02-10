# pyHARM
Python tools for HARM analysis

`pyHARM` is a set of Python functions for analyzing and plotting the output of General-Relativistic Magnetohydrodynamic (GRMHD) simulations.  It includes functions for obtaining a long list of different variables of interest (pressures, temperatures, stress-energy tensor, synchrotron emissivity) based on local fluid state.  It additionally includes reductions for performing surface and volume integrals of various types, and plotting tools, as well as MPI-accelerated scripts for producing movies and time-summed or -averaged reductions over large-scale simulations.

`pyHARM` primarily supports simulations based on the HARM scheme (Gammie et al. 2003) -- [KHARMA](https://github.com/AFD-Illinois/kharma), [iharm3d](https://github.com/AFD-Illinois/iharm3d), and [ebhlight](https://github.com/AFD-Illinois/ebhlight).  It includes Python re-implementations of core parts of the scheme, useful for deriving everything about the simulation not directly present in output files.  It includes limited support for several other codes, either directly or after translation with [EHT-babel](https://github.com/AFD-Illinois/EHT-babel/).

The core of pyHARM is the `FluidDump` object, which behaves similar to a Python dictionary of `numpy` arrays, but calculates its members on the fly by reading the original file and performing operations only as necessary ("lazy" evaluation).  `FluidDump` objects can be sliced similarly to `numpy` arrays, and subsequent file reads and calculations will be done over only the sliced portion.

Finally, as a consequence of needing to read GRMHD output, pyHARM includes definitions of various coordinate systems in Kerr spacetime, as well as tools for dealing with a logically Cartesian grid in various coordinate systems.  These might be independently useful, see `coordinates.py` & `grid.py` if you're looking for just coordinate tools.

## Installing:
The preferred installation method, for flexibility in changing the source as needed, is to run simply:
```bash
$ pip3 install -e .
```
Thereafter pyHARM should be importable from any Python prompt or script run in the same environment.  It can also be installed as a user or system package, but at the cost of less easily modifying the source.

## Examples:
Often, what you need to do is covered somewhere in the existing scripts bundled with pyHARM.  Check out the `scripts/` directory.  The two main scripts are `movie.py` for producing plots and movies, and `analysis.py` for performing a set of reductions over a full simulation's output.  If you need to produce a plot, try adapting something from `pyHARM.plots.figures`.  Adding plots to that directory makes them accessible to `movie.py` when you're ready to run at scale.

For more general usage, the `notebooks` directory has a sample Jupyter notebook playing around with some basic reductions & plots.

## Keys:
The full list of `FluidDump` "keys" is long and growing, and there are some ways to combine keys
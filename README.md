# pyharm
Python tools for HARM analysis

`pyharm` is a set of Python functions for analyzing and plotting the output of General-Relativistic Magnetohydrodynamic (GRMHD) simulations.  It includes functions for obtaining a long list of different variables of interest (pressures, temperatures, stress-energy tensor, synchrotron emissivity) based on local fluid state.  It additionally includes reductions for performing surface and volume integrals of various types, and plotting tools, as well as MPI-accelerated scripts for producing movies and time-summed or -averaged reductions over large-scale simulations.

`pyharm` primarily supports simulations based on the HARM scheme (Gammie et al. 2003) -- [KHARMA](https://github.com/AFD-Illinois/kharma), [iharm3d](https://github.com/AFD-Illinois/iharm3d), and [ebhlight](https://github.com/AFD-Illinois/ebhlight).  It includes Python re-implementations of core parts of the scheme, useful for deriving everything about the simulation not directly present in output files.  It includes limited support for several other codes, either directly or after translation with [EHT-babel](https://github.com/AFD-Illinois/EHT-babel/).

The core of pyharm is the `FluidDump` object, which behaves similar to a Python dictionary of `numpy` arrays, but calculates its members on the fly by reading the original file and performing operations only as necessary ("lazy" evaluation).  `FluidDump` objects can be sliced similarly to `numpy` arrays, and subsequent file reads and calculations will be done over only the sliced portion.

Finally, as a consequence of needing to read GRMHD output, pyharm includes definitions of various coordinate systems in Kerr spacetime, as well as tools for dealing with a logically Cartesian grid in various coordinate systems.  These might be independently useful, see `coordinates.py` & `grid.py` if you're looking for just coordinate tools.

## Installing:
The preferred installation method is to run simply:
```bash
$ pip3 install -e .
```
Thereafter pyharm should be importable from any Python prompt or script run in the same environment.  This command installs `pyharm` as "editable," so that changes to the source will be reflected immediately when next importing the package -- this is recommended, as it is expected that users will have to modify the source code eventually.  `pyharm` can also be installed as a user or system package, but at the cost of less easily modifying the source.

## Examples:
One good starting point learning to use `pyharm` is to try out the existing tools in the `scripts/` directory.  These are quite capable wrappers for large swaths of `pyharm`'s functionality, and are added to the system `$PATH` variable upon installation to make them easy to call.  They support MPI parallelization for use at scale on big clusters (sample batch scripts for some Illinois-relevant clusters are included in `scripts/batch/`).

The two main scripts are `pyharm-movie` for producing plots and movies, and `pyharm-analysis` for performing a set of reductions over a full simulation's output.  If what you need to do involves making a movie or performing a reduction, it can often be implemented as a plugin to these existing scripts -- see the examples in `pyharm.plots.figures` for movies, and in `pyharm.ana.analyses` for reductions & computations.  New additions to those files will automatically become valid arugments to `pyharm-movie` and `pyharm-analysis`.

For more advanced usage, the `notebooks` directory has a sample Jupyter notebook playing around with some basic reductions & plots.

## Keys:
Several parts of `pyharm` try to parse strings to determine behavior, to specify a desired variable, plot, or result from the command line or quickly in notebooks.  These replace what might traditionally be member functions (think `dump.ucov()`) with what acts like a giant Python dictionary (`dump['ucov']`). Each "item" is computed at first access and cached thereafter.

Since it's possible to combine keys in arbitrary ways, there is no master list of items, nor does it make sense to write e.g. `for key in dump` or `if "key" in dump`.  The most reliable way to determine whether something can be computed is to try it, and catch the `ValueError` (for unknown keys) or `IOError` (for known keys not present in a file) if it is not found.

### Base keys (variables):

All the fluid primitives in HARM-like schemes are supported, of course: `rho`, `u`, `u1`, `u2`, `u3`, `uvec`, `B1`, `B2`, `B3`, `bvec`. The `*vec` versions return all 3 components of the primitives.

The contravariant and covariant 4-vectors `ucon`, `ucov`, `bcon`, `bcov` and dot product `bsq` are supported, as well as any individual component of either vector in any coordinate system: native 1,2,3 coordinates `u^{1,2,3}`, KS coordinates `u^{r,th,phi}`, or (soon) Cartesian `u^{x,y,z}`.

Pressures are denoted `Pg` for ideal gas pressure, `Pb` for magnetic pressure, `Ptot` for both summed.  For dumps including output of the 4-current (& eventually for others, via a full step calculation), we additionally define `jcon`, `jcov`, `jsq`, and `Jsq` -- with the former being the 4-current magnitude squared, and the latter the 3-current.

There are a number of specialized or particular definitions more easily defined than described.  See `pyharm.ana.variables` for definitions.  The patterns in that file should be straightforward if/when you want to add your own variables, which will then be accessible from FluidDump objects, and consequently from the command line or notebooks when plotting.

### Properties:
All of a dump's parameters (e.g. the contents of `header` in the Illinois format) are parsed and included in the dictionary `dump.params`.  Anything in this dictionary can be accessed via the shorthand `dump['key']` just like a variable.  These include at least the `gam`, the coordinate system parameters `a`, `hslope`, `r_in`, `r_out`, etc., and the grid size `n1`, `n2`, `n3` to name some of the most useful.  The full list is accessible as `dump.params.keys()`.

### Grid parameters
Finally, members of the `Grid` object are also accessible through `FluidDump` objects, e.g. `dump['gcon']` for the full contravariant metric or `dump['gdet']` for the sqare root of the negative metric determinant.
### Combining elements:

Common mathematical unary operations can just be combined into a variable name to get the result.  This is most common/useful when specifying what to plot, most commonly `log_` for the log base 10.  Similarly, `abs_`, `sqrt_`, and `ln_` for natural log are available.  So for example, asking for `sqrt_bsq` is equivalent to asking for the field magnitude `b`.  You can, of course, perform these operations for yourself in code -- these are present to help you when specifying a variable name on the command line, or for functions which operate only on some small portion of a dump, where the variable needn't be computed over the whole dump.

There are a few modifiers specific to requesting plots which build on this "unary operator" scheme but are mostly *after* any relevant variable. The exception is `symlog_`, which plots a signed value on a symmetric log scale.  The list of postfix options can be gleaned from `plots/frame.py` but includes at least:
* `_array` for plots in native coordinates on the Cartesian logical grid
* `_poloidal` or `_2d`, `_toroidal` for restricting plotting to just one slice or plotting 2D simulation output
* `_1d` likewise for radial/1D simulation output
* `_ghost` for plotting ghost zones (works best with `_array`)
* `_simple` to turn off axis labels, ticks, and colorbars for a presentation or outreach movie, or if you just need the screen real-estate

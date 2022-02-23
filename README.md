# 'dev' branch NOTE:
This branch is significantly changed from previous versions of pyharm.  Therefore, some scripts and even outer portions of the library are temporarily broken until imports and interface changes are updated.  The core features described in the new README sections below should work, as should anything around `movie.py`, i.e. plotting of KHARMA dumps and everything to do with coordinates.  I've tried to update the documentation as I go, but pieces, especially in the open documentation pages, are several versions behind still.

Once everything is in fairly complete working order, the plan is to upload pyharm to pyPI to have a single working base that's easy to install into different environments.  If you'd like to write your own scripts around KHARMA dumps, the file `pyharm/io/kharma.py`, specifically `read_var`, is a good place to start.  For a basic reader, you can safely ignore index operations related to ghost zones and slicing, but note the matrix transformations required to read KHARMA "meshblocks" into a single whole "mesh."

# pyharm
Python tools for HARM analysis

`pyharm` is a set of Python functions for analyzing and plotting the output of General-Relativistic Magnetohydrodynamic (GRMHD) simulations.  It includes functions for obtaining a long list of different variables of interest (pressures, temperatures, stress-energy tensor, synchrotron emissivity) based on local fluid state.  It additionally includes reductions for performing surface and volume integrals of various types, and plotting tools, as well as MPI-accelerated scripts for producing movies and time-summed or -averaged reductions over large-scale simulations.

`pyharm` primarily supports simulations based on the HARM scheme (Gammie et al. 2003) -- [KHARMA](https://github.com/AFD-Illinois/kharma), [iharm3d](https://github.com/AFD-Illinois/iharm3d), and [ebhlight](https://github.com/AFD-Illinois/ebhlight).  It includes Python re-implementations of core parts of the scheme, useful for deriving everything about the simulation not directly present in output files.  It includes limited support for several other codes, either directly or after translation with [EHT-babel](https://github.com/AFD-Illinois/EHT-babel/).

The core of pyharm is the `FluidDump` object, which behaves similar to a Python dictionary of `numpy` arrays, but calculates its members on the fly by reading the original file and performing operations only as necessary ("lazy" evaluation).  `FluidDump` objects can be sliced similarly to `numpy` arrays, and subsequent file reads and calculations will be done over only the sliced portion.

Finally, as a consequence of needing to read GRMHD output, pyharm includes definitions of various coordinate systems in Kerr spacetime, as well as tools for dealing with a logically Cartesian grid in various coordinate systems.  These might be independently useful, see `coordinates.py` & `grid.py` if you're looking for just coordinate tools.

## Installing:
The preferred installation method, for flexibility in changing the source as needed, is to run simply:
```bash
$ pip3 install -e .
```
Thereafter pyharm should be importable from any Python prompt or script run in the same environment.  It can also be installed as a user or system package, but at the cost of less easily modifying the source.

## Examples:
Often, what you need to do is covered somewhere in the existing scripts bundled with pyharm.  Check out the `scripts/` directory.  The two main scripts are `movie.py` for producing plots and movies, and `analysis.py` for performing a set of reductions over a full simulation's output.  If you need to produce a plot, try adapting something from `pyharm.plots.figures`.  Adding plots to that directory makes them accessible to `movie.py` when you're ready to run at scale.

For more general usage, the `notebooks` directory has a sample Jupyter notebook playing around with some basic reductions & plots.

## Keys:
The full list of `FluidDump` "keys" is long and growing, and there are some ways to combine keys, so any list is incomplete.

### Base keys (variables):

All the fluid primitives in HARM-like schemes are supported: rho, `u`, `u1`, `u2`, `u3`, `uvec`, `B1`, `B2`, `B3`, `bvec`. `*vec` versions return all 3 components of the primitives.

The contravariant and covariant 4-vectors `ucon`, `ucov`, `bcon`, `bcov` and dot product `bsq`, as well as any individual component in native 1,2,3 coordinates `u^{1,2,3}`, or in KS coordinates `u^{r,th,phi}`, or (soon) Cartesian `u^{x,y,z}`.

Fluid variables `Pg` for ideal gas pressure, `Pb` for magnetic pressure, `Ptot` for both summed.  

For dumps including output of the 4-current (& soon for others via a full step calculation): `jcon`, `jcov`, `jsq`, `Jsq`

There are a number of specialized or particular definitions which will be easier to define than describe.  See `variables.py` for definitions.  The patterns in that file should be straightforward if/when you want to add your own variables, which will then be accessible from FluidDump objects, and consequently from the command line or notebooks when plotting.

### Combining elements:

Common mathematical unary operations can just be combined into a variable name to get the result.  This is most common/useful when specifying what to plot, most commonly `log_` for the log base 10.  Similarly, `abs_`, `sqrt_`, and `ln_` for natural log are available.  So for example, asking for `sqrt_bsq` is equivalent to asking for the field magnitude `b`.  You can, of course, perform these operations for yourself in code -- however, many scripts take a variable name as argument, e.g. on the command line.  For these, it's very nice to have a few common operations available without writing any code whatsoever.

There are a few modifiers specific to requesting plots which build on this "unary operator" scheme but are mostly *after* any relevant variable. The exception is `symlog_`, which plots a signed value on a symmetric log scale.  The list of postfix options can be gleaned from `plots/frame.py` but includes at least:
* `_array` for plots in native coordinates on the Cartesian logical grid
* `_poloidal` or `_2d`, `_toroidal` for restricting plotting to just one slice or plotting 2D simulation output
* `_1d` likewise for radial/1D simulation output
* `_ghost` for plotting ghost zones (works best with `_array`)
* `_simple` to turn off axis labels, ticks, and colorbars for a presentation or outreach movie, or if you just need the screen real-estate

As mentioned, components of certain vectors (anything suffixed `*cov` or `*con`) can be requested specifically by separating the name and index with `^` or `_`, and their transforms (e.g. to KS coordinates) can be requested similarly by coordinate name.

Anything in the dump parameters or "header," `dump.params.keys()`.  These can be listed for a full view, but include e.g. the adiabatic index `'gam'` and full domain size `'n1','n2','n3'` among a bunch of other things.

Finally, members of the `Grid` object are also accessible through `FluidDump` objects, e.g. `dump['gcon']` for the full contravariant metric or `dump['gdet']` for the sqare root of the negative metric determinant.

## Reductions

See `analysis.py` in scripts and `reductions.py` in pyharm.
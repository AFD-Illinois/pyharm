
.. _keys:

Keys
====

In several places, for flexibility, ``pyharm`` adopts a dictionary-like syntax to refer to what might otherwise be member variables or functions. This allows parsing the dictionary "key" to return a much wider range of items than one might bother to code or cache explicity as traditional member variables/functions.  It also allows flexibility in "passing through" keys: if one object is asked for a key it cannot calculate, it can fall through and ask another object (or several more, in order of preference) -- if any of the objects can furnish the item, it is then returned.

The main place that keys are parsed is when requesting variables or properties calculated from a GRMHD dump file, represented by a :class:`pyharm.fluid_state.FluidState` object.  In addition to performing its own calculations, implemented in :mod:`pyharm.variables`, the ``FluidState`` class can obtain any named items from its component :class:`pyharm.grid.Grid` object or the original dump file (:class:`pyharm.io.interface.DumpFile` subclass).

Even beyond ``FluidState`` objects, both the ``pyharm-movie`` and ``pyharm-analysis`` scripts have additional lists of keys and modifiers that they accept, representing what plots to render or sets of reductions to perform.

Fluid Dump Keys
---------------

All the fluid primitives in HARM-like schemes are supported, of course: ``rho, u, u1, u2, u3, uvec, B1, B2, B3, bvec``. The ``*vec`` versions return all 3 components of the primitives.

The contravariant and covariant 4-vectors ``ucon, ucov, bcon, bcov`` and dot product ``bsq`` are supported, as well as any individual component of either vector in any coordinate system: native 1,2,3 coordinates ``u^{1,2,3}``, KS coordinates ``u^{r,th,phi}``, or (soon) Cartesian ``u^{x,y,z}``.

The contravariant, covariant, and mixed stress-energy and Maxwell tensors ``T`` and ``F`` are supported -- that is, ``T^mu_nu``, ``F^mu^nu``, etc, where ``mu`` and ``nu`` can be any of ``{1,2,3}`` (no automatic coordinate system conversions for tensors yet).  The mixed forms of several portions of ``T`` can be requested separately, notably ``TFl^mu_nu`` for the fluid stress-energy tensor alone, and ``TEM^mu_nu`` for the electromagnetic stress-energy tensor.

Pressures are denoted ``Pg`` for ideal gas pressure, ``Pb`` for magnetic pressure, ``Ptot`` for both summed.  For dumps including output of the 4-current (& eventually for others, via a full step calculation), we additionally define ``jcon, jcov, jsq, Jsq`` -- the lower-case option denotes the 4-current magnitude squared, and the latter the 3-current.

There are a number of specialized or particular definitions more easily defined than described, see :mod:`pyharm.ana.variables` for definitions.  The patterns in that file should be straightforward if/when you want to add your own variables.

Combining elements
~~~~~~~~~~~~~~~~~~
Common mathematical unary operations can just be combined into a variable name to get the result.  This is most common/useful when specifying what to plot, most commonly ``log_`` for the log base 10.  Similarly, ``abs_, sqrt_,`` and ``ln_`` do what one would expect; for example, asking for ``sqrt_bsq`` is equivalent to asking for the field magnitude ``b``.

You can, of course, perform these operations for yourself in code -- these are present to help you when specifying a variable name on the command line, or for functions which operate only on some small portion of a dump, where the variable needn't be computed over the whole dump.

Property Keys
~~~~~~~~~~~~~
All of a dump's parameters (e.g. the contents of `header` in the Illinois format) are parsed and included in the dictionary `dump.params`.  Anything in this dictionary can be accessed via the shorthand `dump['key']` just like a variable.  These include at least the `gam`, the coordinate system parameters `a`, `hslope`, `r_in`, `r_out`, etc., and the grid size `n1`, `n2`, `n3` and zone size `dx1`, `dx2`, `dx3`,  to name some of the most useful.

Properties draw from two dictionaries, which you can list out for yourself: ``dump.params.keys()`` and ``dump.grid.params.keys()``

Grid Keys
~~~~~~~~~
Several members of the :class:`pyharm.grid.Grid` object are also accessible through keys, e.g. ``grid['gcon']`` for the full contravariant metric or ``grid['gdet']`` for the sqare root of the negative metric determinant.  The full list is avaialble by calling :func:`pyharm.grid.Grid.has()`.  Most ``Grid`` quantities have an extra index in front corresponding to the location of the value within a zone -- i.e. .  See :ref:`ref_defs`

Most of the same elements (or specifically, the portions at zone centers, which is usually what's desired) can be accessed from  :class:`pyharm.fluid_state.FluidState` objects as well, e.g. ``dump['gdet']`` to return N3xN2x1 array of metric determinants at zone centers.

Note that 

The same variables are also accessible directly from a :class:`pyharm.grid.Grid`.

See :ref:`ref_coords` for full Grid object documentation.

Plotting Keys
-------------
There are a number of pre-defined figures plotting particular combinations of variables, which can be specified as arguments to ``pyharm-movie``. See :ref:`ref_figures`.

Analysis Keys
-------------
There are also a number of pre-defined sets of reductions, which can be specified as arguments to ``pyharm-analysis``. See :ref:`ref_analyses`.

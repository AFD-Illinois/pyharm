
.. _keys:

Keys
====

Fluid Dump Keys
---------------

All the fluid primitives in HARM-like schemes are supported, of course: ``rho, u, u1, u2, u3, uvec, B1, B2, B3, bvec``. The ``*vec`` versions return all 3 components of the primitives.

The contravariant and covariant 4-vectors ``ucon, ucov, bcon, bcov`` and dot product ``bsq`` are supported, as well as any individual component of either vector in any coordinate system: native 1,2,3 coordinates ``u^{1,2,3}``, KS coordinates ``u^{r,th,phi}``, or (soon) Cartesian ``u^{x,y,z}``.

Pressures are denoted ``Pg`` for ideal gas pressure, ``Pb`` for magnetic pressure, ``Ptot`` for both summed.  For dumps including output of the 4-current (& eventually for others, via a full step calculation), we additionally define ``jcon, jcov, jsq, Jsq`` -- the lower-case option denotes the 4-current magnitude squared, and the latter the 3-current.

There are a number of specialized or particular definitions more easily defined than described, see :mod:`pyharm.ana.variables` for definitions.  The patterns in that file should be straightforward if/when you want to add your own variables.

Combining elements:
~~~~~~~~~~~~~~~~~~~
Common mathematical unary operations can just be combined into a variable name to get the result.  This is most common/useful when specifying what to plot, most commonly ``log_`` for the log base 10.  Similarly, ``abs_, sqrt_,`` and ``ln_`` do what one would expect; for example, asking for ``sqrt_bsq`` is equivalent to asking for the field magnitude ``b``.

You can, of course, perform these operations for yourself in code -- these are present to help you when specifying a variable name on the command line, or for functions which operate only on some small portion of a dump, where the variable needn't be computed over the whole dump.

Property Keys:
~~~~~~~~~~~~~~
All of a dump's parameters (e.g. the contents of `header` in the Illinois format) are parsed and included in the dictionary `dump.params`.  Anything in this dictionary can be accessed via the shorthand `dump['key']` just like a variable.  These include at least the `gam`, the coordinate system parameters `a`, `hslope`, `r_in`, `r_out`, etc., and the grid size `n1`, `n2`, `n3` to name some of the most useful.

Properties draw from two dictionaries, which you can list out for yourself: ``dump.params.keys()`` and ``dump.grid.params.keys()``

Grid Keys
~~~~~~~~~
Several members of the :class:`pyharm.grid.Grid` object are also accessible through keys, e.g. ``grid['gcon']`` for the full contravariant metric or ``grid['gdet']`` for the sqare root of the negative metric determinant.  The full list is avaialble by calling :func:`pyharm.grid.Grid.can_provide()`.  Most ``Grid`` quantities have an extra index in front corresponding to the location of the value within a zone -- i.e. .  See :ref:`ref_defs`

Most of the same elements (or specifically, the portions at zone centers, which is usually what's desired) can be accessed from  :class:`pyharm.fluid_dump.FluidDump` objects as well, e.g. ``dump['gdet']`` to return N3xN2x1 array of metric determinants at zone centers.

Note that 

The same variables are also accessible directly from a :class:`pyharm.grid.Grid`.

See :ref:`ref_coords` for full Grid object documentation.

Plotting Keys
-------------
There are a number of pre-defined figures plotting particular combinations of variables, which can be specified as arguments to ``pyharm-movie``. See :ref:`ref_figures`.

Analysis Keys
-------------
There are also a number of pre-defined sets of reductions, which can be specified as arguments to ``pyharm-analysis``. See :ref:`ref_analyses`.

.. _ref_coords:

Coordinate Tools
================

pyharm's tools for dealing with coordinates are split into two files: :mod:`pyharm.coordinates`, for dealing with general-relativistic quantities (e.g. the metric) and transformations between coordinate systems, and :mod:`pyharm.grid` for holding a Cartesian mesh of native coordinates, coupled with their local metric, connection coefficients, and transformation matrices back to some "base" coordinate system (currently must be Kerr-Schild coordinates)

Coordinate System Classes
-------------------------

Coordinate systems are implemented as classes, each of which subclasses :class:`pyharm.coordinates.CoordinateSystem` and implements the methods it contains:

.. autoclass:: pyharm.coordinates.CoordinateSystem
   :members:

The coordinate systems currently supported are Minkowski (:class:`pyharm.coordinates.Minkowski`), Kerr-Schild (:class:`pyharm.coordinates.KS`), Modified Kerr-Schild (:class:`pyharm.coordinates.MKS`), and "Funky" Modified Kerr-Schild (:class:`pyharm.coordinates.FMKS`).  The ``MKS`` and ``FMKS`` systems are used to concentrate zones toward the midplane, and to make zones near the pole larger, in order to preserve accuracy in simulations while maintaining a reasonable time step (which is driven by the Courant condition at the pole).  Details on these coordinate systems can be found `here <https://github.com/AFD-Illinois/docs/wiki/Coordinates>`_.

The Grid Class
--------------

.. automodule:: pyharm.grid
   :special-members:
   :members:

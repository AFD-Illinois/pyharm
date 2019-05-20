.. module:: pyHARM
.. moduleauthor:: Ben Prather <bprathr2@illinois.edu>

.. _ref_coords:

Coordinate Tools
================

pyHARM's tools for dealing with coordinates are split into two files: :mod:`pyHARM.coordinates`, for dealing with continuous transformations between different coordinate systems in the Kerr metric, and :mod:`pyHARM.grid` for holding a Cartesian mesh of native coordinates, coupled with their local metric, connection coefficients, and transformation matrices back to some "base" coordinate system (currently must be Kerr-Schild coordinates)

Coordinate Systems
------------------

Coordinate systems are implemented as classes, each of which subclasses :class:`pyHARM.coordinates.CoordinateSystem` and implements the methods it contains:

.. autoclass:: pyHARM.coordinates.CoordinateSystem
   :members:

The coordinate systems currently supported are Minkowski (:class:`pyHARM.coordinates.Minkowski`), Kerr-Schild (:class:`pyHARM.coordinates.KS`), Modified Kerr-Schild (:class:`pyHARM.coordinates.MKS`), and "Funky" Modified Kerr-Schild (:class:`pyHARM.coordinates.FMKS`).

The Grid Class
--------------

.. automodule:: pyHARM.grid
   :members:

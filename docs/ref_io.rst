.. _ref_io:

File Reading
============
Mostly when reading files you'll be interacting with the wrapper functions directly in the :mod:`pyharm.io` namespace, documented here.

.. automodule:: pyharm.io
   :members:

File Filters
------------
Behind the general/auto-dispatch functions in :mod:`pyharm.io`, ``pyharm`` uses "filter" classes representing each file with a consistent interface.  The interface itself is documented below, followed by examples for KHARMA and iharm3d files, which have some additional functionality.

Note that while this is left to each filter to decide, the designed & fastest option is to read (nearly) nothing at object creation, reading variables only when they requested with :func:`pyharm.io.interface.read_var`, and caching them for future calls.  In addition, read_var should support returning only a slice of a file, either by reading just the necessary slice (preferable) or by slicing the output array before returning it.


.. automodule:: pyharm.io.interface
   :members:

.. automodule:: pyharm.io.kharma
   :members:

.. automodule:: pyharm.io.iharm3d
   :members:

Additional Utilities
--------------------
Finally, there are separate modules for dealing with the iharm3d header format and for "gridfiles" containing zone locations and metric values, as both are quite common to encounter/read/write on their own.

.. automodule:: pyharm.io.iharm3d_header
   :members:

.. automodule:: pyharm.io.gridfile
   :members:

Variable names in pyHARM
========================

The central mode of interacting with ``FluidDump`` objects is via ``[]``, i.e., Python's ``__getitem__()`` function.  This can be used to request nearly any variable or property of a dump file, so long as one knows under what name it is recorded.

A listing of such names follows.  Any of these can be used in the style of ``dump['named_variable']`` to return a numpy scalar or ndarray corresponding to the desired value.

This list may not always be kept up-to-date -- try reading ``variables.py`` and ``fluid_dump.py``
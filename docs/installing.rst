Installing pyharm
=================

The preferred method of obtaining and installing ``pyharm`` is to run simply:
::

    $ git clone https://github.com/AFD-Illinois/pyharm.git
    $ cd pyharm/
    $ pip3 install -e . 

(If ``pip3`` is not found, just use ``pip``, which *usually* points to the same thing.)

Thereafter, ``pyharm`` should be importable from any Python prompt, notebook, or script run in the same environment.  This command installs ``pyharm`` as "editable," so that changes to the source will be reflected immediately when next importing the package -- this is recommended, as it is expected that users will want to make some modifications to the library code as they extend it for their uses.  ``pyharm`` can also be installed as a user or system package, but at the cost of less easily modifying the source.

``pyharm`` also provides an environment file for the Anaconda package manager.  While the ``pip`` package is instabllable to an Anaconda environment, Anaconda packages may be faster & more up to date.  To create & install to a new Anaconda environment, run the following:

::

    $ conda env create -f environment.yml
    $ conda activate pyharm
    $ pip3 install -e .

MPI
---

The scripts `pyharm-movie` and `pyharm-analysis` can be parallelized across several nodes of a cluster using MPI.  This requires additionally installing `mpi4py`, which is not listed as a hard dependency since this is not core functionality for all systems.

Note that since Anaconda ships its own version of MPI which conflicts with system installations, it is advised to install `mpi4py` through `pip` rather than `conda`.  Further note that ``pyharm`` does not support Intel's `IMPI` due to some weirdness with `mpi4py`'s tasking infrastructure -- on Frontera, this means using the module `mvapich2-x` instead.
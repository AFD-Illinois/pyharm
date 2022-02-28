Installing pyharm
=================

The supported/main method for installation is with the Anaconda Python distribution/package manager.  Just download it from [here](https://www.anaconda.com/distribution/#download-section), make sure that ``conda`` is in your ``PATH`` variable, and run the following:

::

    $ conda env create -f environment.yml
    $ conda activate pyharm
    $ python3 setup.py develop

The HARM algorithm portion of the code (and eventually, pieces of the analysis) are accelerated with [loopy](https://mathema.tician.de/software/loopy/) and [pyopencl](https://mathema.tician.de/software/pyopencl/), which can take advantage of OpenCL compute devices like graphics cards or specialized CPU libraries.  If you want to use an OpenCL device, copy its ICD file into Anaconda's environment, e.g. for Nvidia on Linux:

::

    $ cp /etc/OpenCL/vendors/nvidia.icd /path/to/anaconda3/envs/pyharm/etc/OpenCL/vendors/

pyharm can then be run or imported by entering the ``conda`` environment:

::

    $ conda activate pyharm
    $ python3
    >> import pyharm

or

::

    $ conda activate pyharm
    $ python3 pyharm/harm.py

In general, ``pyharm`` should more or less act like an installable ``setuptools`` package -- however, user and system installs are not tested so YMMV.

Installing without OpenCL
=========================

If you just want the analysis tools, and don't care about the OpenCL code, pyharm can be installed without OpenCL using just the command:

::

    $ python setup.py develop

If you wish to install the OpenCL-based dependencies without anaconda (this isn't well supported/tested), you can try the following:

::

    $ pip3 install --user pyopencl[pocl]
    $ git clone https://github.com/inducer/loopy.git; cd loopy
    $ python3 setup.py install --user
# pyHARM
Python tools for HARM: algorithm and analysis

pyHARM is a set of Python functions implementing (most of) the current version of the HARM scheme for general-relativistic magnetohydrodynamics (GRMHD) from Gammie et al 2003, used by e.g. `iharm3d`, `bhlight`, and others.  It also includes a set of analysis and plotting tools built on these functions, for reading and analyzing HDF5 output from the C-based harm codes, or translated from other codes with [EHT-babel](https://github.com/AFD-Illinois/EHT-babel/).

## Installing:
The supported/main method for installation is with anaconda.  Just download it from [here](https://www.anaconda.com/distribution/#download-section), make sure that `conda` is in your `PATH`, and run the following:

```bash
$ conda env create -f environment.yml
$ conda activate pyHARM
$ python3 setup.py develop
```

The HARM algorithm portion of the code (and eventually, pieces of the analysis) are accelerated with [loopy](https://mathema.tician.de/software/loopy/) and [pyopencl](https://mathema.tician.de/software/pyopencl/), which can take advantage of OpenCL compute devices like graphics cards or specialized CPU libraries.  If you want to use an OpenCL device, copy its ICD file into Anaconda's environment, e.g. for Nvidia on Linux:

```bash
$ cp /etc/OpenCL/vendors/nvidia.icd /path/to/anaconda3/envs/pyHARM/etc/OpenCL/vendors/
```

### Alternate install

If for some reason you don't (or can't) use Anaconda, issue the following commands someplace where you keep libraries & source code:

```bash
$ pip3 install --user pyopencl[pocl]
$ git clone https://github.com/inducer/loopy.git; cd loopy
$ python3 setup.py install --user
```

Then add pyHARM from its own directory once you have added dependencies:

```bash
$ cd /path/to/pyHARM
$ python setup.py develop
```

In general, `pyHARM` should more or less act like an installable `setuptools` package -- however, YMMV as this is not tested well.

## Running the HARM algorithm:
tl;dr

```bash
$ python3 pyHARM/harm.py -p problemname
```

Where `problemname` is a subdirectory of `prob/`, with one python file of the same name and one file named param.dat.  The former should define a function:

```python
def init(params, G, P):
```

which takes a dictionary of parameter names `int` `float` or `str` with values, a Grid object constructed from the parameters, and the array of primitives P to which it should write.  The current problems give a good idea of what's possible right now.


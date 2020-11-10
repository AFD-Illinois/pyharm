# pyHARM
Python tools for HARM analysis

pyHARM is a set of Python functions for analyzing and plotting the output of General-Relativistic Magnetohydrodynamic (GRMHD) simulations, including functions for derivation of a number of different quantities (pressure, stress-energy tensor, etc) from the simulation's output, reductions taking account  


The primary target is simulations based on the HARM scheme (Gammie et al. 2003) -- [iharm3d](https://github.com/AFD-Illinois/iharm3d), [ebhlight](https://github.com/AFD-Illinois/ebhlight), [KHARMA](https://github.com/AFD-Illinois/kharma), and others.  It includes Python re-implementations of most of the scheme, useful for deriving everything about the simulation not directly present in output files.  Output from certain other codes can be converted to a readable format with [EHT-babel](https://github.com/AFD-Illinois/EHT-babel/).

## Installing:
The preferred installation method, for flexibility in changing the source as needed, is to run simply:
```bash
$ python3 setup.py develop
```
Thereafter pyHARM should be importable from any Python prompt or script run in the same environment.  It can also be installed as a user or system package with `pip`.


There is also an included Anaconda environment for users who would prefer Anaconda versions of the dependencies -- also note that any future (optional!) OpenCL integration will require the Anaconda environment.

```bash
$ conda env create -f environment.yml
$ conda activate pyHARM
$ python3 setup.py develop
```

## Examples:
The `notebooks` directory has a sample Jupyter notebook playing around with 

The shorter scripts in the `scripts/` directory are fairly idiomatic examples of current `pyHARM` tools and usage.

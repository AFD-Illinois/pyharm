#!/bin/bash

# Currently can't use TACC python...
module unload python3 phdf5

source activate pyHARM

export OPENCL_VENDOR_PATH=~/libs/anaconda3/envs/pyHARM/etc/OpenCL/vendors
export PYOPENCL_CTX=1

if [[ $1 == "numa" ]]; then
echo "Running NUMA"
numactl --interleave=all python3 src/harm.py -p mhdmodes --use_ctypes=1
else
echo "Running normal"
python3 src/harm.py -p mhdmodes --profile=1 --use_ctypes=1 --debug=1 --P_U_P_test=1
fi

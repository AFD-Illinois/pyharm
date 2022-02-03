

import sys
import h5py

import matplotlib.pyplot as plt

from pyHARM.ana.results import *

var = sys.argv[1]
ivar = sys.argv[2]
files = sys.argv[3:]

plt.figure(figsize=(12,6))

for fil in files:
    infil = h5py.File(fil, "r")
    name = fil.replace("/eht_out.h5","")
    if 't' in ivar:
        ivt, vt = get_result(infil, ivar, var, mesh=False)
        if len(ivt.shape) > 1:
            iv = ivt[0][:,0]
            v = np.array([np.sum(vt[i,:]) for i in range(vt.shape[0])])
        else:
            iv = ivt
            v = vt
        plt.plot(iv, v, label=name)
    else:
        plt.plot(*get_result(infil, ivar, var), label=name)
    plt.yscale('log')
    plt.xlim((0, 10000))

plt.legend()
plt.savefig("compare_"+var+"_"+ivar+".png")

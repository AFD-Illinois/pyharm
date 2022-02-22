

import sys
import h5py

import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

from pyHARM.ana_results import *
from pyHARM.reductions import pspec

ivar = sys.argv[1]
var = sys.argv[2]
files = sys.argv[3:]

r_at = 3

plt.figure(figsize=(12,6))

for i, fil in enumerate(files):
    name = fil.replace("/eht_out.h5","")
    col = "C{}".format(i+1)
    if 't' in ivar:
        ivr, vr = get_result(fil, ivar, var, mesh=False)
        if len(ivr.shape) > 1:
            iv = ivr[0][:,0]
            v = np.array([vr[i,i_of(ivr[1][0,:], r_at)] for i in range(vr.shape[0])])
        else:
            iv = ivr
            v = vr
    else:
        iv, v = get_result(fil, ivar, var)
    iv, v = iv[np.nonzero(v)], v[np.nonzero(v)]
    if "log_" in ivar:
        plt.semilogx(iv, v, label=name)
        plt.xlim((2000, 30000))
        plt.ylim(0,2)
    else:
        plt.plot(iv[:-1], v[:-1], label=name, color=col)
        plt.xlim((0, 30000))
        #plt.ylim(-2.5,2)
    print(name,"average 6k+ is: ", np.mean(v[i_of(iv,6000):]))
    #print(name,"average 12k+ is: ", np.mean(v[i_of(iv,12000):]))
    v_sub = v[i_of(iv,6000):i_of(iv,10000)-1]
    print(name,"norm var 6k-10k", np.std(v_sub) / np.mean(v_sub))
    print(name,"rms var 6k-10k", np.sqrt(np.mean(v_sub)**2 + np.std(v_sub)**2) / np.mean(v_sub))
    v_sub = v[i_of(iv,2000):i_of(iv,8000)]
    print(name,"norm var 2k-8k", np.std(v_sub) / np.mean(v_sub))
    print(name,"rms var 2k-8k", np.sqrt(np.mean(v_sub)**2 + np.std(v_sub)**2) / np.mean(v_sub))

    t_fit = 5000
    powerfit = Polynomial.fit(iv[i_of(iv, t_fit):], v[i_of(iv, t_fit):], 1) #, w=iv[i_of(iv, t_fit):])
    print(powerfit)
    #plt.plot(iv[i_of(iv, t_fit):], powerfit(iv)[i_of(iv, t_fit):], label=name+" fit", color=col, linestyle='--')

plt.legend()
plt.savefig("compare_"+ivar+"_"+var+".png")

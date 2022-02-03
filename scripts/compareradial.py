

import sys
import h5py

import matplotlib.pyplot as plt

from pyHARM.ana.results import *
from pyHARM.util import i_of

ivar = sys.argv[1]
var = sys.argv[2]
files = sys.argv[3:]

plt.figure(figsize=(12,6))

for fil in files:
    name = fil.replace("/eht_out.h5","")
    if 't' in ivar:
        if var in ["T_jet", "K_jet"]:
            gam = get_header_var(fil, 'gam')
            ivt, rho = get_result(fil, ivar, "rho_jet", mesh=False)
            _, p = get_result(fil, ivar, "Pg_jet", mesh=False)
            if var == "T_jet":
                vt = p/(gam - 1)/rho
            elif var == "K_jet":
                vt = p/rho**(gam)
        else:
            try:
                ivt, vt = get_result(fil, ivar, var, mesh=False)
            except KeyError:
                continue
            if "^" in var:
                vt = np.abs(vt)
        # Get the average ourselves, i.e. plot r
        iv = ivt[1][0,:]
        t = ivt[0][:,0]
        v = np.mean(vt[i_of(t, 6000):i_of(t, 10000),:], axis=0)
        plt.loglog(iv, v, label=name)
        print("Run ", name)
        print("Value 10M: {}".format(np.mean(v[i_of(iv, 10)])))
        print("Value 100M: {}".format(np.mean(v[i_of(iv, 100)])))

    else:
        plt.loglog(*get_result(fil, ivar, var), label=name)
    plt.xlim()

if var == "Area_jet":
    plt.loglog(iv, v[0]*(iv**2), 'k--', label="r^2")
elif var == "Mdot_jet":
    plt.loglog(iv, v[0]*(iv), 'k--', label="r^1")
elif "P_" in var:
    plt.loglog(iv, v[-1]*np.ones_like(iv), 'k--', label="flat")
else:
    # Usual downward/flat guidelines
    if v[0] < 0:
        plt.loglog(iv, v[-1]*(iv[-1])/(iv), 'k--', label="r^-1")
        plt.loglog(iv, v[-1]*(iv[-1]**2)/(iv**2), 'k--', label="r^-2")
        plt.loglog(iv, v[-1]*(iv[-1]**3)/(iv**3), 'k--', label="r^-3")
    else:
        plt.loglog(iv, 2*v[0]/(iv), 'k--', label="r^-1")
        plt.loglog(iv, 2*v[0]/(iv**2), 'k--', label="r^-2")
        plt.loglog(iv, 2*v[0]/(iv**3), 'k--', label="r^-3")

    if var in ["T_jet", "K_jet"]:
        plt.loglog(iv, iv/iv, 'k--', label="=1")
        plt.loglog(iv, 3*iv/iv, 'k--', label="=3")

plt.legend()
plt.suptitle(var + " vs r, averaged")
plt.savefig("compare_"+var+"_"+ivar+".png")

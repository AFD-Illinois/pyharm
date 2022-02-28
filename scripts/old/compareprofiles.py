

import sys
import h5py

import matplotlib.pyplot as plt

from pyharm.ana_results import *
from pyharm.util import i_of

x = 5
var = sys.argv[1]
ivar = sys.argv[2]
files = sys.argv[3:]

plt.figure(figsize=(12,6))

for fil in files:
    try:
        infil = h5py.File(fil, "r")
    except OSError as e:
        infil = h5py.File(fil+"/eht_out.h5", "r")
    name = fil.replace("/eht_out.h5","")
    if 'r' in ivar:
        if var in ["T_jet", "K_jet"]:
            gam = get_header_var(infil, 'gam')
            iv, rho = get_result(infil, ivar, "rho_jet", mesh=False)
            _, p = get_result(infil, ivar, "Pg_jet", mesh=False)
            if var == "T_jet":
                v = p/(gam - 1)/rho
            elif var == "K_jet":
                v = p/rho**(gam)
        else:
            try:
                iv, v = get_result(infil, ivar, var, mesh=False)
            except KeyError:
                continue
            if "^" in var:
                v = np.abs(v)
        # plot th at r=X
        r = iv[0][:,0]
        iv = iv[1][i_of(r, x),:]
        v = v[i_of(r, x),:]
        plt.plot(iv, v, label=name)
        print("Run ", name)
        print("Value upper pole: {}".format(v[i_of(iv, 0)]))
        print("Value upper funnel: {}".format(v[i_of(iv, np.pi/6)]))
        print("Value lower pole: {}".format(v[i_of(iv, np.pi)]))
        print("Value lower funnel: {}".format(v[i_of(iv, 5*np.pi/6)]))
    else:
        plt.plot(*get_result(infil, ivar, var), label=name)
    plt.xlim()

plt.legend()
plt.suptitle(var + " vs th, r={}".format(x))
plt.savefig("compare_{}_th_r={}.png".format(var,x))

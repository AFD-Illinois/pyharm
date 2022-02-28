

import sys
import h5py
import psutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyharm.ana_results import *
from pyharm.util import run_parallel

n_avg = 10

var = sys.argv[1]
ivar = sys.argv[2]
files = sys.argv[3:]

infils = []
for fil in files:
    try:
        infils.append(h5py.File(fil, "r"))
    except OSError as e:
        infils.append(h5py.File(fil+"/eht_out.h5", "r"))

nframes = len(get_ivar(infils[1], 't'))
frame_dir = "frames_" + var + "_" + ivar
os.makedirs(frame_dir, exist_ok=True)

def plot(n):
    plt.figure(figsize=(12,6))
    print("Making frame {}".format(n))
    for fil,infil in zip(files,infils):
        name = fil.replace("/eht_out.h5","")
        if var in ["T_jet", "K_jet"]:
            ivt, rho = get_result(infil, ivar, "rho_jet", mesh=False)
            _, p = get_result(infil, ivar, "Pg_jet", mesh=False)
            if var == "T_jet":
                vt = p/(5/3 - 1)/rho
            elif var == "K_jet":
                vt = p/rho**(5/3)
        else:
            ivt, vt = get_result(infil, ivar, var, mesh=False)
        # Get the average ourselves, i.e. plot r
        iv = ivt[1][n*n_avg,:]
        v = np.mean(vt[n*n_avg:(n+1)*n_avg], axis=0)
        plt.loglog(iv, v, label=name)
        plt.xlim()

    plt.loglog(iv, 2*v[0]/(iv**2), 'k--', label="-2")

    plt.legend()
    plt.savefig(os.path.join(frame_dir, 'frame_%08d.png' % n))

if __name__ == "__main__":
    nthreads = psutil.cpu_count()
    run_parallel(plot, nframes//n_avg, nthreads)

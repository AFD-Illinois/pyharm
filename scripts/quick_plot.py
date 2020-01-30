################################################################################
#                                                                              #
#  PLOT ONE PRIMITIVE                                                          #
#                                                                              #
################################################################################

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyHARM.ana.iharm_dump import IharmDump
import pyHARM.ana.plot as bplt
from pyHARM.ana.units import get_units_M87

# TODO parse these instead of hard-coding
USEARRSPACE = True

SIZE = 15
#window = (0, SIZE, 0, SIZE)
window = (-SIZE, SIZE, -SIZE, SIZE)
# window=(-SIZE/4, SIZE/4, 0, SIZE)
pdf_window = (-10, 0)
FIGX = 10
FIGY = 10

dumpfile = sys.argv[1]
var = sys.argv[2]
# Optionally take extra name, otherwise just set it to var
name = sys.argv[-1]

if len(sys.argv) > 4:
    munit = float(sys.argv[4])
    cgs = get_units_M87(munit)
    print("Uisng M_unit: ", munit)
    unit = cgs[sys.argv[3]]
    print("Will multiply by unit {} with value {}".format(sys.argv[3], unit))
    name = var + "_units"
else:
    unit = 1

dump = IharmDump(dumpfile, add_jcon=True)

# Manually do PDFs. TODO add all reductions by name somehow
if var.split("/")[0] == "pdf":
    var_og = var.split("/")[1]
    d_var, d_var_bins = np.histogram(np.log10(dump[var_og]), bins=200, range=pdf_window,
                                                  # Weights have to be the same shape as var
                                                  weights=np.repeat(dump['gdet'], dump.N3).reshape(dump[var_og].shape),
                                                  density=True)
    fig = plt.figure(figsize=(FIGX, FIGY))
    plt.plot(d_var_bins[:-1], d_var)
    plt.title("PDF of " + var_og)
    plt.xlabel("Log10 value")
    plt.ylabel("Probability")

    plt.savefig(var_og + "_pdf.png", dpi=100)
    plt.close(fig)
    exit()

# Plot vectors in 4-pane layout
fig = plt.figure(figsize=(FIGX, FIGY))

if var in ['jcon', 'jcov', 'ucon', 'ucov', 'bcon', 'bcov']:
    axes = [plt.subplot(2, 2, i) for i in range(1, 5)]
    for n in range(4):
        bplt.plot_xy(axes[n], dump, np.log10(dump[var][n] * unit), arrayspace=USEARRSPACE, window=window)
else:
    # TODO allow specifying vmin/max, average from command line or above
    ax = plt.subplot(1, 1, 1)
    bplt.plot_xy(ax, dump, dump[var] * unit, log=False, arrayspace=USEARRSPACE, window=window)

plt.tight_layout()
plt.savefig(name + "_xy.png", dpi=100)
plt.close(fig)

# Plot XZ
fig = plt.figure(figsize=(FIGX, FIGY))

if var in ['jcon', 'jcov', 'ucon', 'ucov', 'bcon', 'bcov']:
    axes = [plt.subplot(2, 2, i) for i in range(1, 5)]
    for n in range(4):
        bplt.plot_xz(axes[n], dump, np.log10(dump[var][n] * unit), arrayspace=USEARRSPACE, window=window)
else:
    ax = plt.subplot(1, 1, 1)
    bplt.plot_xz(ax, dump, dump[var] * unit, log=False, arrayspace=USEARRSPACE, window=window)

plt.tight_layout()

plt.savefig(name + "_xz.png", dpi=100)
plt.close(fig)

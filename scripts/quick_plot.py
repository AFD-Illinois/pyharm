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

import pyharm
import pyharm.plots.plot_dumps as pplt
from pyharm import pretty
from pyharm.units import get_units_M87

FIGX = 10
FIGY = 10

# TODO parse these instead of hard-coding
USEARRSPACE = True
SUM = False

if USEARRSPACE:
    window = (0, 1, 0, 1)
else:
    SIZE = 50
    #window = (0, SIZE, 0, SIZE)
    window = (-SIZE, SIZE, -SIZE, SIZE)
    # window=(-SIZE/4, SIZE/4, 0, SIZE)

pdf_window = (-10, 0)

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

dump = pyharm.load_dump(dumpfile)

if dump['n3'] > 1:
    fig = plt.figure(figsize=(FIGX, FIGY))
    plt.title(pretty(var))
    # Plot XY
    # Plot vectors in 4-pane layout
    if var in ['jcon', 'jcov', 'ucon', 'ucov', 'bcon', 'bcov']:
        axes = [plt.subplot(2, 2, i) for i in range(1, 5)]
        for n in range(4):
            pplt.plot_xy(axes[n], dump, dump[var][n] * unit, native=USEARRSPACE, window=window)
    else:
        # TODO allow specifying vmin/max, average from command line or above
        ax = plt.subplot(1, 1, 1)
        pplt.plot_xy(ax, dump, dump[var] * unit, log=False, native=USEARRSPACE, integrate=SUM, window=window)

    plt.tight_layout()
    plt.savefig(name + "_xy.png", dpi=100)
    plt.close(fig)

# Plot XZ
fig = plt.figure(figsize=(FIGX, FIGY))
plt.title(pretty(var))

if var in ['jcon', 'jcov', 'ucon', 'ucov', 'bcon', 'bcov']:
    axes = [plt.subplot(2, 2, i) for i in range(1, 5)]
    for n in range(4):
        pplt.plot_xz(axes[n], dump, dump[var][n] * unit, native=USEARRSPACE, window=window)
else:
    ax = plt.subplot(1, 1, 1)
    pplt.plot_xz(ax, dump, dump[var] * unit, log=False, native=USEARRSPACE, integrate=SUM, window=window)

plt.tight_layout()

plt.savefig(name + "_xz.png", dpi=100)
plt.close(fig)

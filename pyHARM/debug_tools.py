# Various debugging tools: mapping areas of the grid, checks for NaNs, etc.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def check_nan_prims(params, P):
    # Print the surrounding areas of the first 10 NaNs found
    # TODO allow to specify one prim to check, slice to check
    nidxs = np.argwhere(np.isnan(P))
    if nidxs.shape[0] > 10:
        nidxs = nidxs[:10]
    for idx in nidxs:
        print_area_prims(params, P, idx)


def check_nan_array(arr, slc=None):
    if slc is not None:
        nidxs = np.argwhere(np.isnan(arr))
    else:
        nidxs = np.argwhere(np.isnan(arr[slc]))
    print(nidxs)
    if nidxs.shape[0] > 10:
        nidxs = nidxs[:10]
    for idx in nidxs:
        print_area_array(arr, idx)


def print_area_prims(params, P, idx):
    # Print surrounding zones of a location idx in form [prim_number, i, j, k]
    dbg_slice = P[idx[0], idx[1]-1:idx[1]+1, idx[2]-1:idx[2]+1, idx[3]-1:idx[3]+1]
    print(params['var_names'][idx[0]], ":", dbg_slice)


def print_area_array(arr, idx):
    dbg_slice = arr[idx[0]-1:idx[0]+1, idx[1]-1:idx[1]+1, idx[2]-1:idx[2]+1]
    print(dbg_slice)


def shape_of_nonzero_portion(arr, tag=""):
    for i, idx_list in enumerate(np.nonzero(arr)):
        print("{} axis {} {}:{}".format(tag, i, np.min(idx_list), np.max(idx_list)))


names_plotted = {}
def plot_var(G, var, name, slc=None):
    s = G.slices
    if not isinstance(var, np.ndarray):
        var = var.get()

    if slc is None:
        if len(var.shape) == 3:
            slc = (s.b, s.b, G.NG)
        elif len(var.shape) == 2:
            slc = (s.b, s.b)
        elif len(var.shape) == 2:
            slc = (s.b,)
        else:
            raise ValueError("Dimension of array unsupported for plotting!")

    # Automatic numbering
    if name not in names_plotted:
        names_plotted[name] = 1
    else:
        names_plotted[name] += 1

    fig, ax = plt.subplots()
    mesh = plt.pcolormesh(var[slc])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mesh, cax=cax)
    plt.savefig(name+str(names_plotted[name])+".png")
    plt.close(fig)


def plot_prims(G, P, name, slc=None):
    for i in range(P.shape[0]):
        plot_var(G, P[i], name + str(i) + "_", slc)

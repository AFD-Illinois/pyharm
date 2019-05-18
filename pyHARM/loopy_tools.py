# Tools for compiling loopy kernels

import numpy as np
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

from pyHARM.defs import Var

# TODO param stuff like this for easy testing?
lsize = (32, 1, 1)
otags = ("l.0", "g.0", "g.1")
#itags = ("l.0", "l.1", "l.2")
itags = ("unr", "unr", "unr")


def make_run_kernel():
    pass


def spec_prims_kernel(knl, shape=None, ng=None):
    if shape is not None:
        knl = lp.fix_parameters(knl, nprim=int(shape[0]), n1=int(shape[1]), n2=int(shape[2]), n3=int(shape[3]))
    if ng is not None:
        knl = lp.fix_parameters(knl, ng=ng)
    return knl


def tune_prims_kernel(knl, shape=None, ng=None, prefetch_args=()):
    """Parameters for 3D"""
    knl = spec_prims_kernel(knl, shape, ng)
    # This assumes linear, see above
    knl = lp.split_iname(knl, "k", lsize[0], outer_tag=otags[0], inner_tag=itags[0])
    knl = lp.split_iname(knl, "j", lsize[1], outer_tag=otags[1], inner_tag=itags[1])
    knl = lp.split_iname(knl, "i", lsize[2], outer_tag=otags[2], inner_tag=itags[2])
    knl = lp.set_loop_priority(knl, "p,i_outer,j_outer,k_outer,i_inner,j_inner,k_inner")
    for arg in prefetch_args:
        knl = lp.add_prefetch(knl, arg, "i_inner,j_inner,k_inner", default_tag="l.auto")

    #knl = lp.set_options(knl, no_numpy=True)
    return knl


def tune_grid_kernel(knl, shape=None, ng=None, prefetch_args=()):
    """Parameters for 3D"""
    if shape is not None:
        if len(shape) > 3:
            knl = lp.fix_parameters(knl, ndim=int(shape[0]), n1=int(shape[1]), n2=int(shape[2]), n3=int(shape[3]))
        else:
            knl = lp.fix_parameters(knl, n1=int(shape[0]), n2=int(shape[1]), n3=int(shape[2]))
    if ng is not None:
        knl = lp.fix_parameters(knl, ng=ng)
    knl = lp.split_iname(knl, "k", lsize[0], outer_tag=otags[0], inner_tag=itags[0])
    knl = lp.split_iname(knl, "j", lsize[1], outer_tag=otags[1], inner_tag=itags[1])
    knl = lp.split_iname(knl, "i", lsize[2], outer_tag=otags[2], inner_tag=itags[2])
    for arg in prefetch_args:
        knl = lp.add_prefetch(knl, arg, "i_inner,j_inner,k_inner", default_tag="l.auto")

    #knl = lp.set_options(knl, no_numpy=True)
    return knl


def tune_geom_kernel(knl, shape=None, ng=None):
    """Parameters specific to kernels where one array is 2D and one is 3D"""
    # These kernels get run on lots of grid sizes
    if shape is not None:
        pass # Deal with various vectros/tensors
        #knl = lp.fix_parameters(knl, n1=int(shape[0]), n2=int(shape[1]), n3=int(shape[2]))
    knl = lp.split_iname(knl, "k", lsize[0], outer_tag="for", inner_tag="unr") # Worthwhile to cache so explicitly?
    #knl = lp.split_iname(knl, "k", lsize[0], outer_tag=otags[0], inner_tag=itags[0])
    knl = lp.split_iname(knl, "j", lsize[1], outer_tag=otags[0], inner_tag=itags[0])
    knl = lp.split_iname(knl, "i", lsize[2], outer_tag=otags[1], inner_tag=itags[1])

    # TODO caching geom values is especially important. Try things here...
    if shape is not None:
        if len(shape) > 4:
            knl = lp.prioritize_loops(knl, "mu, nu,i_outer,i_inner,j_outer,j_inner,k_outer,k_inner")
        elif len(shape) > 3:
            knl = lp.prioritize_loops(knl, "mu,i_outer,i_inner,j_outer,j_inner,k_outer,k_inner")
        else:
            knl = lp.prioritize_loops(knl, "i_outer,i_inner,j_outer,j_inner,k_outer,k_inner")

    # Currently these functions get used on frontend, too
    #knl = lp.set_options(knl, no_numpy=True)

    return knl


def vecArrayArgs(*names, ghosts=True):
    args = []
    for name in names:
        if ghosts:
            shape_sym = "(ndim, n1 + 2*ng, n2 + 2*ng, n3 + 2*ng)"
        else:
            shape_sym = "(ndim, n1, n2, n3)"
        space = lp.AddressSpace.GLOBAL
        args.append(lp.ArrayArg(name, dtype=np.float64, shape=shape_sym, address_space=space))
    return args


def primsArrayArgs(*names, ghosts=True, dtype=np.float64):
    args = []
    for name in names:
        if ghosts:
            shape_sym = "(nprim, n1 + 2*ng, n2 + 2*ng, n3 + 2*ng)"
        else:
            shape_sym = "(nprim, n1, n2, n3)"
        space = lp.AddressSpace.GLOBAL
        args.append(lp.ArrayArg(name, dtype=dtype, shape=shape_sym, address_space=space))
    return args


def scalarArrayArgs(*names, ghosts=True, dtype=np.float64):
    args = []
    for name in names:
        if ghosts:
            shape_sym = "(n1 + 2*ng, n2 + 2*ng, n3 + 2*ng)"
        else:
            shape_sym = "(n1, n2, n3)"
        space = lp.AddressSpace.GLOBAL
        args.append(lp.ArrayArg(name, dtype=dtype, shape=shape_sym, address_space=space))
    return args


def gscalarArrayArgs(*names, ghosts=True, dtype=np.float64):
    args = []
    for name in names:
        if ghosts:
            shape_sym = "(n1 + 2*ng, n2 + 2*ng)"
        else:
            shape_sym = "(n1, n2)"
        space = lp.AddressSpace.GLOBAL
        args.append(lp.ArrayArg(name, dtype=dtype, shape=shape_sym,
                                address_space=space, offset=lp.auto))
    return args


def gvectorArrayArgs(*names, ghosts=True, dtype=np.float64):
    args = []
    for name in names:
        if ghosts:
            shape_sym = "(ndim, n1 + 2*ng, n2 + 2*ng)"
        else:
            shape_sym = "(ndim, n1, n2)"
        space = lp.AddressSpace.GLOBAL
        args.append(lp.ArrayArg(name, dtype=dtype, shape=shape_sym,
                                address_space=space, offset=lp.auto))
    return args


def gtensorArrayArgs(*names, ghosts=True, dtype=np.float64):
    args = []
    for name in names:
        if ghosts:
            shape_sym = "(ndim, ndim, n1 + 2*ng, n2 + 2*ng)"
        else:
            shape_sym = "(ndim, ndim, n1, n2)"
        space = lp.AddressSpace.GLOBAL
        args.append(lp.ArrayArg(name, dtype=dtype, shape=shape_sym,
                                address_space=space, offset=lp.auto))
    return args

# TODO friends don't let friends write their own preprocessors
# These are pretty straightforward cases for using pymbolic
def replace_prim_names(code):
    for var in Var:
        code = code.replace(var.name, str(var.value))
    return code

# TODO I so bet there's a loopy function for iname offsets. I so bet that.
def add_ghosts(code):
    for var in ["i", "j", "k"]:
        for token_before in ["[", ",", ", ", "("]:
            # Plus must be first, as everything will match the rule after adding +ng!
            for token_after in ["+", "-", " -", " +", "]", ")", ",", " ,"]:
                code = code.replace(token_before + var + token_after, token_before + var + "+ng" + token_after)
    return code

# Parse parameters files and sys.argv

import os
import sys
import numpy as np


def defaults():
    return {# Basics & 
            'version': "pyHARM-alpha-0.01",
            'ng': 3,
            'outdir': ".",
            'paramfile': "param.dat",
            'gridfile': "grid.h5",
            'n_prims_passive': 0,
            # Stability
            'dt_start': 1.0e-06,
            'safe_step_increase': 1.5,
            'cour': 0.9,
            'dt_static': False,
            'sigma_max': 100.0,
            'u_over_rho_max': 100.0,
            'gamma_max': 25.0,
            # Inversion stability
            'invert_err_tol': 1.e-8,
            'invert_iter_max': 8,
            'invert_iter_delta': 1.e-5,
            'debug': 0,
            'profile': 0
            }


def parse_dat(params, fname):
    """Parse the HARM params.dat format to produce a Python dict.
    params.dat format:
    [tag] name = value
    Where tag is in {dbl, float, int, str} corresponding to desired datatype.
    All lines not in this format are ignored, though conventionally comments begin with '# '
    """
    fp = open(fname, "r")
    for line in fp:
        # Trim out trailing newline, anything after '#', stray parentheses or extra spaces
        ls = [token.strip('()') for token in line[:-1].split("#")[0].split(" ") if token != '']
        # And blank lines
        if len(ls) == 0:
            continue
        # Parse according to tag
        if ls[0] in ["[dbl]", "[float]"]:
            params[ls[1]] = float(ls[-1])
        elif ls[0] == "[int]":
            params[ls[1]] = int(ls[-1])
        elif ls[0] == "[str]":
            params[ls[1]] = str(ls[-1])
    return params

def parse_parthenon_dat(params, fname):
    """Parse the KHARMA params.dat format to produce a Python dict.
    params.dat format:
    <header/subheader>
    name = value
    All lines not in this format are ignored, though conventionally comments begin with '# '
    """
    fp = open(fname, "r")
    for line in fp:
        # Trim out trailing newline, anything after '#', stray parentheses, headers
        ls = [token.strip().strip('()') for token in line[:-1].split("#")[0].split("<")[0].split("=") if token != '']
        # And blank lines
        if len(ls) == 0:
            continue
        # Parse, assuming int->float->str
        try:
            if "." in ls[-1]:
                params[ls[0]] = float(ls[-1])
            else: 
                params[ls[0]] = int(ls[-1])
        except ValueError:
            params[ls[0]] = ls[-1]
    
    # Now do any repairs specific to translating the Parthenon->iharm3d naming scheme
    for pair in (#('nx1','n1'), ('nx2','n2'), ('nx3','n3'),
                 ('x1min', 'startx1'), ('x2min', 'startx2'), ('x3min', 'startx3'),
                 ('gamma', 'gam')):
        params[pair[1]] = params[pair[0]]

    params['r_in'] = np.exp(params['x1min'])
    params['r_out'] = np.exp(params['x1max'])
    if 'a' in params:
        params['r_eh'] = (1. + np.sqrt(1. - params['a'] ** 2))

    params['dx1'] = (params['x1max'] - params['x1min'])/params['nx1']
    params['dx2'] = (params['x2max'] - params['x2min'])/params['nx2']
    params['dx3'] = (params['x3max'] - params['x3min'])/params['nx3']

    params['n_prim'] = params['np'] = 8 # Parthenon stores extra variables under new names
    params['n_dim'] = 4
    # I frequently omit these expecting defaults
    if not 'mks_smooth' in params:
        params['mks_smooth'] = 0.5
    if not 'poly_xt' in params:
        params['poly_xt'] = 0.82
    if not 'poly_alpha' in params:
        params['poly_alpha'] = 14.0

    if "cartesian" in params['base']:
        params['coordinates'] = "cartesian"
    elif "fmks" in params['transform'] or "funky" in params['transform']:
        params['coordinates'] = "fmks"
        params['poly_norm'] = 0.5 * np.pi * 1. / (1. + 1. / (params['poly_alpha'] + 1.) *
                                            1. / np.power(params['poly_xt'], params['poly_alpha']))
    elif "mks" in params['transform'] or "modif" in params['transform']:
        params['coordinates'] = "mks"
    else:
        print("Defaulting coordinate system...")
        params['coordinates'] = "fmks"

    return params

def fix(params):
    # Fix common parameter mistakes
    if ('derefine_poles' in params) and (params['derefine_poles'] == 1):
        params['metric'] = "fmks"
    if 'Rout' in params:
        params['r_out'] = params['Rout']
    if 'electrons' in params and params['electrons'] == 1:
        params['electrons'] = True
        params['n_prim'] = 10
        params['prim_names'] = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3", "KTOT", "KEL"]
    else:
        params['electrons'] = False
        params['n_prim'] = 8
        params['prim_names'] = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3"]
    if 'metric' in params and 'coordinates' not in params:
        params['coordinates'] = params['metric'].lower()


    return params


def parse_argv(params, argv):
    """Parse the usual command line arguments.  Note -p can also specify problem name"""
    if "-o" in argv:
        params['outdir'] = argv[argv.index("-o")+1]
    if "-p" in argv:
        # Take -p as "param" or "problem" since it's easy
        # Detect on the fly which we're dealing with
        param_path = argv[argv.index("-p")+1]
        param_altpath = os.path.join(sys.path[0], "../prob/"+param_path+"/param.dat")
        if os.path.isfile(param_path):
            params['paramfile'] = param_path
        elif os.path.exists(param_altpath):
            params['paramfile'] = param_altpath
        else:
            raise ValueError("Parameter file not found!")

    return params


def override_from_argv(params, argv):
    """Allow passing any parameter as argument overriding the datfile"""
    for parm in argv:
        if parm[:2] == "--":
            if "=" in parm:
                kv = parm[2:].split("=")
                params[kv[0]] = kv[1]
            else:
                params[parm[2:]] = argv[argv.index(parm)+1]
    return params
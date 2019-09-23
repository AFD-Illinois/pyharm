# Parse parameters files and sys.argv

import os
import sys


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
        # Trim out trailing newline, anything after '#', stray parenthesis or extra spaces
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
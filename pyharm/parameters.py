__license__ = """
 File: parameters.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020-2022, AFD Group at UIUC
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np

__doc__ = \
"""Parse and handle parameters.
These functions exist primarily to standardize reading simulation parameters
from a number of sources, and name them ~generally the same things.
"""

def parse_iharm3d_dat(params, fname):
    """Parse the iharm3d params.dat file format to produce a Python dict.
    The params.dat format is:

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
    return fix(params)

def parse_parthenon_dat(string):
    """Parse the Parthenon/KHARMA params.dat format to produce a Python dict.
    The run.par format is:

    | <header/subheader>
    | name = value

    All lines not in this format are ignored, though conventionally comments begin with '# '.
    This implementation does not distinguish packages (<>), nor line continuations (&) correctly.
    """
    # TODO:
    # parse & include headers, so we can guarantee picking out the correct nx1/2/3
    # parse lists and line continuations (&) correctly

    params = {}

    # Things KHARMA will never use/modify but need to be *something* for IL HDF file header
    params['version'] = "kharma-alpha-0.1"
    params['gridfile'] = "NONE"
    params['n_prims_passive'] = 0

    for line in string.split("\n"):
        # Trim out trailing newline, anything after '#', stray parentheses, headers
        ls = [token.strip().strip('()') for token in line.split("#")[0].split("<")[0].split("=") if token != '']
        # And blank lines
        if len(ls) == 0:
            continue
        # Parse, assuming float->int->str and taking the largest surviving numbers (to avoid block-specific nxN)
        try:
            if "." in ls[-1]:
                if ls[0] not in params or float(params[ls[0]]) < float(ls[-1]):
                    params[ls[0]] = float(ls[-1])
            else:
                if ls[0] not in params or int(params[ls[0]]) < int(ls[-1]):
                    params[ls[0]] = int(ls[-1])
        except ValueError:
            params[ls[0]] = ls[-1]

    # Translate coordinate naming scheme
    # TODO will there be exp/superexp w/BL? Likely not, but...
    if "cartesian" in params['base']:
        params['coordinates'] = "cartesian"
    elif "fmks" in params['transform'] or "funky" in params['transform']:
        params['coordinates'] = "fmks"
    elif "mks" in params['transform'] or "modif" in params['transform']:
        params['coordinates'] = "mks"
    elif "eks" in params['transform'] or "exponent" in params['transform']:
        params['coordinates'] = "eks"
    elif "null" in params['transform'] and "ks" in params['base']:
        params['coordinates'] = "ks"
    elif "null" in params['transform'] and "bl" in params['base']:
        params['coordinates'] = "bl"
    elif "super" in params['transform']:
        params['coordinates'] = "superexp"
    else:
        print("Defaulting KHARMA coordinate system to fmks...")
        params['coordinates'] = params['metric'] = "fmks"

    return fix(params)


def fix(params):
    """Fix a bunch of common problems and omissions in parameters dictionaries.
    Already called by both parameter parsing functions.
    """

    # Now do any repairs specific to translating the Parthenon->iharm3d naming scheme
    for pair in (('nx1','n1'), ('nx2','n2'), ('nx3','n3'),
                 ('n1','n1tot'), ('n2','n2tot'), ('n3','n3tot'),
                 ('x1min', 'startx1'), ('x2min', 'startx2'), ('x3min', 'startx3'),
                 ('gamma', 'gam'), ('dt', 'dump_cadence'), ('tlim', 'tf'), ('cfl', 'cour')):
        if (pair[0] in params):
            params[pair[1]] = params[pair[0]]

    if (not 'r_out' in params) and 'Rout' in params:
        params['r_out'] = params['Rout']

    if not ('prim_names' in params):
        if 'electrons' in params and params['electrons']:
            params['electrons'] = True # In case it was an int
            params['prim_names'] = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3", "KTOT", "KEL"]
        else:
            params['electrons'] = False
            params['prim_names'] = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3"]

    if 'n_prim' not in params:
        if 'n_prims' in params: # This got messed up *often*
            params['n_prim'] = params['n_prims']
        elif 'prim_names' in params:
            params['n_prim'] = len(params['prim_names'])

    # Lots of layers of legacy coordinate specification
    # To be clear the modern way is to use the 'coordiantes' key,
    # as one of usually 'fmks','mks','cartesian' (or 'eks', 'mks3', 'bhac_mks', etc, see grid.py)
    if 'coordinates' not in params:
        if 'metric' in params:
            # The old 'metric' key was uppercase
            params['coordinates'] = params['metric'].lower()
        elif ('derefine_poles' in params) and (params['derefine_poles'] == 1):
            # Very old iharm3d specified derefine_poles
            params['coordinates'] = "fmks"
        else:
            print("Defaulting to MKS coordinate system...")
            params['coordinates'] = "mks"

    # Also add eh radius
    if params['coordinates'] != "cartesian" and 'r_eh' not in params and 'a' in params:
        params['r_eh'] = (1. + np.sqrt(1. - params['a'] ** 2))

    # Metric defaults we're pretty safe in setting
    # TODO add these to coordinates.py
    if params['coordinates'] == "fmks":
        if not 'mks_smooth' in params:
            params['mks_smooth'] = 0.5
        if not 'poly_xt' in params:
            params['poly_xt'] = 0.82
        if not 'poly_alpha' in params:
            params['poly_alpha'] = 14.0
        if 'poly_norm' not in params:
            params['poly_norm'] = 0.5 * np.pi * 1. / (1. + 1. / (params['poly_alpha'] + 1.) *
                                        1. / np.power(params['poly_xt'], params['poly_alpha']))

    # If we must guess r_in and/or set coordinate stuff, do it last
    if 'r_in' not in params:
        if 'x1min' not in params:
            params['r_in'] = np.exp((params['n1'] * np.log(params['r_eh']) / 5.5 - np.log(params['r_out'])) /
                                    (-1. + params['n1'] / 5.5))
        else:
            params['r_in'] = np.exp(params['x1min'])

    if 'x1min' not in params:
        params['x1min'] = np.log(params['r_in'])
    if 'x1max' not in params:
        params['x1max'] = np.log(params['r_out'])

    if 'x2min' not in params:
        params['x2min'] = 0.
        if params['coordinates'] in ('fmks', 'mks', 'cartesian',):
            params['x2max'] = 1.0
        else:
            params['x2max'] = np.pi

    if 'x3min' not in params:
        params['x3min'] = 0.
        if params['coordinates'] in ('cartesian',):
            params['x3max'] = 1.0
        else:
            params['x3max'] = 2*np.pi

    if 'dx1' not in params:
        params['dx1'] = (params['x1max'] - params['x1min']) / params['n1']
        params['dx2'] = (params['x2max'] - params['x2min']) / params['n2']
        params['dx3'] = (params['x3max'] - params['x3min']) / params['n3']

    # Translate anything we added -> iharm3d again
    # TODO pick one form geez
    for pair in (('x1min', 'startx1'), ('x2min', 'startx2'), ('x3min', 'startx3')):
        if (pair[0] in params):
            params[pair[1]] = params[pair[0]]    

    # Defualts we never changed but didn't always record
    if 'gam_e' not in params:
        params['gam_e'] = 4./3
    if 'gam_p' not in params:
        params['gam_p'] = 5./3

    return params

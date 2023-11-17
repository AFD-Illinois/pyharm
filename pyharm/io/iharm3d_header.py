__license__ = """
 File: iharm3d_header.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020-2023, Ben Prather and AFD Group at UIUC
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
import h5py, sys

from ..grid import Grid
from .. import parameters

__doc__ = \
"""Functions for reading and writing the 'header/' portion of iharm3d/Illinois HDF files.
Lots of writing and validation tools, hence splitting from the filter and restart file classes.
"""

# FORMAT SPEC
# Constants corresponding to the HARM format as written in HDF5.  Useful for checking compliance,
# and writing files correctly
# Keys in the base directory
base_keys = ['t', 'dt', 'dump_cadence', 'full_dump_cadence', 'is_full_dump', 'n_dump', 'n_step']
# Keys that should be in the header:
header_keys = ['cour', 'gam', 'tf', 'gridfile', 'metric', 'reconstruction', 'version', 'prim_names',
               'n1', 'n2', 'n3', 'n_prim', 'n_prims_passive']
header_keys_electrons = ['cour', 'gam', 'tf', 'fel0', 'gam_e', 'gam_p', 'tptemax', 'tptemin',
                         'gridfile', 'metric', 'reconstruction', 'version', 'prim_names',
                         'n1', 'n2', 'n3', 'n_prim', 'n_prims_passive',
                         'has_electrons', 'has_radiation']
# Keys in header/geom
geom_keys = ['dx1', 'dx2', 'dx3', 'startx1', 'startx2', 'startx3', 'n_dim']
# Keys in potential geom/mks
mks_keys = ['r_eh', 'r_in', 'r_out', 'a', 'hslope']
mmks_keys = ['poly_alpha', 'poly_xt']
fmks_keys = ['mks_smooth', 'poly_alpha', 'poly_xt']

# Defaults for some things we only write so ipole does not get mad
defaults = {'n_prims_passive': 0, 'version': "pyharm-writer-0", 'gridfile': 'none', 'tf': 30000, 'cour': 0.9, 'reconstruction': 'weno5'}

# Translations of things the rest of pyharm tolerates/understands as elements of 'params', but which are not
# conformant for headers and should be written with an alternate name.
translations = {'n1': 'n1tot', 'n2': 'n2tot', 'n3': 'n3tot',
                'metric': 'coordinates', 'metric_run': 'coordinates'}

def write_hdr(params, outf):
    """Write a valid iharm3d/Illinois HDF header"""
    if 'electrons' in params and params['electrons']:
        _write_param_grp(params, header_keys_electrons, 'header', outf)
    else:
        _write_param_grp(params, header_keys, 'header', outf)
    
    G = Grid(params)
    geom_params = {'dx1': G.dx[1],
                   'dx2': G.dx[2],
                   'dx3': G.dx[3],
                   'startx1': G.startx[1],
                   'startx2': G.startx[2],
                   'startx3': G.startx[3],
                   'n_dim': 4}

    _write_param_grp(geom_params, geom_keys, 'geom', outf['header'])


    if params['coordinates'] in ["cartesian", "minkowski"]:
        # No special geometry to record
        pass
    elif params['coordinates'] == "mks":
        _write_param_grp(params, mks_keys, 'mks', outf['header/geom'])
    elif params['coordinates'] == "fmks":
        _write_param_grp(params, mks_keys+fmks_keys, 'fmks', outf['header/geom'])
        # FOR NOW: Duplicate into "mmks" header because codes expect things there
        _write_param_grp(params, mks_keys+fmks_keys, 'mmks', outf['header/geom'])
    elif params['coordinates'] == "mmks":
        _write_param_grp(params, mks_keys+mmks_keys, 'mmks', outf['header/geom'])
    else:
        raise NotImplementedError("Fluid dump files in {} coordinates not implemented!".format(params['coordinates']))

    # Write everything except the pointers to an archival copy
    if "extras" not in outf:
        outf.create_group("extras")
    if "extras/pyharm_params" not in outf:
        outf['extras'].create_group("pyharm_params")
    for key in params:
        try:
            _write_value(outf, params[key], 'header/'+key)
        except TypeError:
            # This particular key is expected to be missing, keep quiet
            if key != 'phdf_aux':
                print("Not writing parameter {}".format(key), file=sys.stderr)

def read_hdr(grp):
    """Read an iharm3d/Illinois HDF header.
    :parameter params: Dictionary of parameters to update. Otherwise, read_hdr will create an empty one
    """
    # Handle lots of different calling conventions
    if isinstance(grp, str):
        fil = h5py.File(grp, "r")
        grp = fil['header']
        close_file = True
    elif 'header' in grp:
        grp = grp['header']
        close_file = False
    else:
        close_file = False
    
    params = {}
    try:
        # Scoop all the keys that are data, leave the sub-groups
        for key in [key for key in list(grp) if isinstance(grp[key], h5py.Dataset)]:
            params[key] = grp[key][()]

        # The 'geom' group should contain at most one more layer of sub-group
        geom = grp['geom']
        for key in geom:
            if isinstance(geom[key], h5py.Dataset):
                params[key] = geom[key][()]
            else:
                for sub_key in geom[key]:
                    params[sub_key] = geom[key+'/'+sub_key][()]

    except KeyError as e:
        print("Warning: {}".format(e))
        print("File is older than supported, but pyharm will attempt to continue".format(e))


    # Fix up all the nasty non-Unicode strings
    _decode_all(params)

    # EXTRAS AND WORKAROUNDS
    # Turn the version string into components
    if 'version' not in params:
        params['version'] = "iharm-alpha-3.6"
        print("Unknown version: defaulting to {}".format(params['version']))

    if "KORAL" in params['version']:
        params['codename'] = "KORAL"
        params['vnum'] = params['version'].split("v")[1]
    else:
        # Illinois versioning scheme
        params['codename'], params['codestatus'], params['vnum'] = params['version'].split("-")
    
    # Split vnum into a list of each point-separated number
    params['vnum'] = [int(x) for x in params['vnum'].split(".")]

    # iharm3d-specific workarounds:
    if params['codename'] == "iharm":
        # Work around naming bug before output v3.4
        if params['vnum'] < [3, 4]:
            names = []
            for name in params['prim_names'][0]:
                names.append(name)
            params['prim_names'] = names

        # Work around bad radius names before output v3.6
        if ('r_in' not in params) and ('Rin' in params):
            params['r_in'], params['r_out'] = params['Rin'], params['Rout']

    # Renames we've done for pyharm  or are useful
    # params_key -> likely name in existing header
    # header_key -> desired name in conformant/pyharm header
    for params_key in translations:
        header_key = translations[params_key]
        if params_key in params:
            if isinstance(params[params_key], str):
                params[header_key] = params[params_key].lower()
            else:
                params[header_key] = params[params_key]

    if close_file:
        fil.close()

    return parameters.fix(params)


def hdf5_to_dict(h5grp):
    """Recursively load group contents into nested dictionaries.
    Used to load analysis output while keeping shapes straight.
    """
    do_close = False
    if isinstance(h5grp, str):
        h5grp = h5py.File(h5grp, "r")
        do_close = True

    ans = {}
    for key, item in h5grp.items():
        if isinstance(item, h5py._hl.group.Group):
            # Call recursively
            ans[key] = hdf5_to_dict(h5grp[key])
        elif isinstance(item, h5py._hl.dataset.Dataset):
            # Otherwise read the dataset
            ans[key] = item[()]

    if do_close:
        h5grp.close()

    # This runs the un-bytes-ing too much, but somehow not enough
    _decode_all(ans)
    return ans


def dict_to_hdf5(wdict, h5grp):
    """Write nested dictionaries of Python values to HDF5 groups nested within the group/file h5grp
    If a filename is specified, automatically opens/closes the file.
    """
    do_close = False
    if isinstance(h5grp, str):
        h5grp = h5py.File(h5grp, "r")
        do_close = True

    for key, item in wdict.items():
        if isinstance(item, dict):
            # Call recursively
            if not key in h5grp:
                h5grp.create_group(key)
            dict_to_hdf5(wdict[key], h5grp[key])
        else:
            # Otherwise write the value to a dataset
            _write_value(h5grp, wdict[key], key)

    if do_close:
        h5grp.close()


def _decode_all(bytes_dict):
    """HDF5 strings are read as bytestrings.  This converts nested dicts of bytestrings to native UTF-8"""
    for key in bytes_dict:
        # Decode bytes/numpy bytes
        if isinstance(bytes_dict[key], (bytes, np.bytes_)):
            bytes_dict[key] = bytes_dict[key].decode('UTF-8')
        # Split ndarray of bytes into list of strings
        elif isinstance(bytes_dict[key], np.ndarray):
            if bytes_dict[key].dtype.kind == 'S':
                bytes_dict[key] = [el.decode('UTF-8') for el in bytes_dict[key]]
        # Recurse for any subfolders
        elif isinstance(bytes_dict[key], dict):
            _decode_all(bytes_dict[key])

    return bytes_dict

def _write_value(outf, value, name):
    """Write a single value to HDF5 file outf, automatically converting Python3 strings & lists"""
    if isinstance(value, list):
        if isinstance(value[0], str):
            load = [np.array(n.upper().encode("ascii", "ignore"), dtype='S20') for n in value]
        else:
            load = value
    elif isinstance(value, str):
        load = np.array(value.upper().encode("ascii", "ignore"), dtype='S20')
    else:
        load = value

    # If key exists just overwrite it
    if name not in outf:
        outf[name] = load
    else:
        outf[name][()] = load

def _write_param_grp(params, key_list, name, parent):
    if not name in parent:
        parent.create_group(name)
    outgrp = parent[name]
    for key in key_list:
        if key in translations.keys():
            _write_value(outgrp, params[translations[key]], key)
        elif key in params:
            _write_value(outgrp, params[key], key)
        elif key in defaults:
            _write_value(outgrp, defaults[key], key)
        else:
            print("WARNING: Format specifies writing key {} to {}, but not present!".format(key, outgrp.name))
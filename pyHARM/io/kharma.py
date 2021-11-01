
import numpy as np
import glob
import h5py

import pyHARM.parameters as parameters
from pyHARM.grid import Grid
from .phdf import phdf

def read_hdr(fname, params=None):
    params_was_none = False
    if params is None:
        params = {}
        params_was_none = True

    # TODO fall back to actual file as well?
    with h5py.File(fname, "r") as infile:
        if 'Input' in infile:
            file_string = infile['Input'].attrs['File']
            if type(file_string) != str:
                file_string = file_string.decode("UTF-8")
            parameters.parse_parthenon_dat(file_string, params)
        else:
            raise RuntimeError("No parameters could be found in KHARMA dump {}".format(fname))

    # But if params had elements, assume it was taken care of by the user and don't require a file
    return params

def get_dump_time(fname):
    dfile = h5py.File(fname, 'r')

    if 'Info' in dfile.keys():
        t = dfile['Info'].attrs['Time']
    else:
        t = 0

    dfile.close()
    return t

def read_dump(fname, add_ghosts=False, params=None):
    f = phdf(fname)

    params = read_hdr(fname, params)
    G = Grid(params)

    # First, quit if we've been asked for ghost zones but can't produce them
    if add_ghosts and not f.IncludesGhost:
        raise ValueError("Ghost zones aren't available in file {}".format(fname))

    # This is the number of ghost zones present in the file
    ng_f = f.IncludesGhost * f.NGhost

    if add_ghosts:
        params['ng'] = f.NGhost
        # This is the number we should include on output
        ng_ix = ng_f
    else:
        params['ng'] = 0
        # This is the number we should include on output
        ng_ix = 0

    # Trivial dimensions don't have ghosts
    if params['n2'] == 1:
        ng_iy = 0
    else:
        ng_iy = ng_ix

    if params['n3'] == 1:
        ng_iz = 0
    else:
        ng_iz = ng_ix

    # Set incidental parameters from what we've read
    params['t'] = f.Time
    params['n_step'] = f.NCycle
    # Add dump number if we've conformed to usual naming scheme
    fname_parts = fname.split("/")[-1].split(".")
    if len(fname_parts) > 2:
        params['n_dump'] = int(fname_parts[2])
    params['startx1'] = G.startx[1]
    params['startx2'] = G.startx[2]
    params['startx3'] = G.startx[3]
    params['dx1'] = dx = G.dx[1]
    params['dx2'] = dy = G.dx[2]
    params['dx3'] = dz = G.dx[3]

    # Lay out the blocks and determine total mesh size
    bounds = []
    for ib in range(f.NumBlocks):
        bb = f.BlockBounds[ib]
        # Internal location of the block i.e. starting/stopping indices in the final, big mesh
        bound = [int((bb[0]+dx/2 - G.startx[1])/dx)-ng_ix, int((bb[1]+dx/2 - G.startx[1])/dx)+ng_ix,
                 int((bb[2]+dy/2 - G.startx[2])/dy)-ng_iy, int((bb[3]+dy/2 - G.startx[2])/dy)+ng_iy,
                 int((bb[4]+dz/2 - G.startx[3])/dz)-ng_iz, int((bb[5]+dz/2 - G.startx[3])/dz)+ng_iz]
        bounds.append(bound)



    if 'prims.Ktot' in f.Variables:
        # If any electrons are present
        params['n_prim'] = 13
        params['prim_names'] = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3", "KTOT", "KEL_KAWAZURA", "KEL_WERNER", "KEL_ROWAN", "KEL_SHARMA"]
        # Optionally allocate enough space for ghost zones
        P = np.zeros((13, G.NTOT[1]+2*ng_ix, G.NTOT[2]+2*ng_iy, G.NTOT[3]+2*ng_iz))
    else:
        P = np.zeros((8, G.NTOT[1]+2*ng_ix, G.NTOT[2]+2*ng_iy, G.NTOT[3]+2*ng_iz))

    # Read blocks into their slices of the full mesh
    for ib in range(f.NumBlocks):
        b = bounds[ib]
        # Exclude ghost zones if we don't want them
        if ng_ix == 0 and ng_f != 0:
            if params['n2'] == 1:
                o = [ng_f, -ng_f, None, None, None, None]
            elif params['n3'] == 1:
                o = [ng_f, -ng_f, ng_f, -ng_f, None, None]
            else:
                o = [ng_f, -ng_f, ng_f, -ng_f, ng_f, -ng_f]
        else:
            o = [None, None, None, None, None, None]

        # Try reading new fancy format
        if 'prims.rho' in f.Variables:
            # False == don't flatten into 1D array
            rho_array = f.Get('prims.rho', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1]].transpose(2,1,0)
            P[0, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = rho_array
            del rho_array
            u_array = f.Get('prims.u', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1]].transpose(2,1,0)
            P[1, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = u_array
            del u_array
            uvec_array = f.Get('prims.uvec', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1],:].transpose(3,2,1,0)
            P[2:5, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = uvec_array
            del uvec_array
            if 'prims.B' in f.Variables:
                B_array = f.Get('prims.B', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1],:].transpose(3,2,1,0)
                P[5:8, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = B_array
                del B_array
            if 'prims.Ktot' in f.Variables:
                k_array = f.Get('prims.Ktot', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1]].transpose(2,1,0)
                P[8, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = k_array
                del k_array
            # Not part of the spec.  Read independently (read all these independently?)
            # if 'prims.Kel_Howes' in f.Variables:
            #     k_array = f.Get('prims.Kel_Howes', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1]].transpose(2,1,0)
            #     P[9, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = k_array
            #     del k_array
            if 'prims.Kel_Kawazura' in f.Variables:
                k_array = f.Get('prims.Kel_Kawazura', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1]].transpose(2,1,0)
                P[9, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = k_array
                del k_array
            if 'prims.Kel_Werner' in f.Variables:
                k_array = f.Get('prims.Kel_Werner', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1]].transpose(2,1,0)
                P[10, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = k_array
                del k_array
            if 'prims.Kel_Rowan' in f.Variables:
                k_array = f.Get('prims.Kel_Rowan', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1]].transpose(2,1,0)
                P[11, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = k_array
                del k_array
            if 'prims.Kel_Sharma' in f.Variables:
                k_array = f.Get('prims.Kel_Sharma', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1]].transpose(2,1,0)
                P[12, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = k_array
                del k_array
        else:
            # Old file formats
            prim_array = f.Get('c.c.bulk.prims', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1],:].transpose(3,2,1,0)
            if prim_array.shape[0] == 8:
                P[:, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = prim_array
            else:
                P[:5, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = prim_array
                if 'c.c.bulk.B_prim' in f.Variables:
                    P[5:, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = \
                        f.Get('c.c.bulk.B_prim', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1],:].transpose(3,2,1,0)

    return (P, params)

def read_jcon(fname, add_ghosts=False):
    # TODO take params and don't re-read etc
    f = phdf(fname)

    params = read_hdr(fname)
    G = Grid(params)

    # First, quit if we've been asked for ghost zones but can't produce them
    if add_ghosts and not f.IncludesGhost:
        raise ValueError("Ghost zones aren't available in file {}".format(fname))

    # This is the number of ghost zones present in the file
    ng_f = f.IncludesGhost * f.NGhost

    if add_ghosts:
        params['ng'] = f.NGhost
        # This is the number we should include on output
        ng_ix = ng_f
    else:
        params['ng'] = 0
        # This is the number we should include on output
        ng_ix = 0

    # Trivial dimensions don't have ghosts
    if params['n2'] == 1:
        ng_iy = 0
    else:
        ng_iy = ng_ix

    if params['n3'] == 1:
        ng_iz = 0
    else:
        ng_iz = ng_ix

    dx = G.dx[1]
    dy = G.dx[2]
    dz = G.dx[3]

    # Lay out the blocks and determine total mesh size
    bounds = []
    for ib in range(f.NumBlocks):
        bb = f.BlockBounds[ib]
        # Internal location of the block i.e. starting/stopping indices in the final, big mesh
        bound = [int((bb[0]+dx/2 - G.startx[1])/dx)-ng_ix, int((bb[1]+dx/2 - G.startx[1])/dx)+ng_ix,
                 int((bb[2]+dy/2 - G.startx[2])/dy)-ng_iy, int((bb[3]+dy/2 - G.startx[2])/dy)+ng_iy,
                 int((bb[4]+dz/2 - G.startx[3])/dz)-ng_iz, int((bb[5]+dz/2 - G.startx[3])/dz)+ng_iz]
        bounds.append(bound)

    # Optionally allocate enough space for ghost zones
    jcon = np.zeros((4, G.NTOT[1]+2*ng_ix, G.NTOT[2]+2*ng_iy, G.NTOT[3]+2*ng_iz))

    # Read blocks into their slices of the full mesh
    for ib in range(f.NumBlocks):
        b = bounds[ib]
        # Exclude ghost zones if we don't want them
        if ng_ix == 0 and ng_f != 0:
            if params['n2'] == 1:
                o = [ng_f, -ng_f, None, None, None, None]
            elif params['n3'] == 1:
                o = [ng_f, -ng_f, ng_f, -ng_f, None, None]
            else:
                o = [ng_f, -ng_f, ng_f, -ng_f, ng_f, -ng_f]
        else:
            o = [None, None, None, None, None, None]
        # False == don't flatten into 1D array
        jcon[:, b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = \
            f.Get('jcon', False)[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1],:].transpose(3,2,1,0)

    return jcon

def read_psi_cd(fname, add_ghosts=False):
    return read_scalar(fname, 'prims.psi_cd', add_ghosts=add_ghosts)

def read_divb(fname, add_ghosts=False):
    divb = read_scalar(fname, 'divB', add_ghosts=add_ghosts)
    if divb is None:
        divb = read_scalar(fname, 'divB_ct', add_ghosts=add_ghosts)
    if divb is None:
        divb = read_scalar(fname, 'divB_cd', add_ghosts=add_ghosts)
    return divb

def read_fail_flags(fname, add_ghosts=False):
    return read_scalar(fname, 'pflag', dtype=np.int32, add_ghosts=add_ghosts).astype(int)

def read_floor_flags(fname, add_ghosts=False):
    return read_scalar(fname, 'fflag', dtype=np.int32, add_ghosts=add_ghosts).astype(int) // 32

def read_scalar(fname, scalar_name, dtype=np.float64, add_ghosts=False):
    f = phdf(fname)

    params = read_hdr(fname)
    G = Grid(params)

    # First, quit if we've been asked for ghost zones but can't produce them
    if add_ghosts and not f.IncludesGhost:
        raise ValueError("Ghost zones aren't available in file {}".format(fname))

    # This is the number of ghost zones present in the file
    ng_f = f.IncludesGhost * f.NGhost

    if add_ghosts:
        params['ng'] = f.NGhost
        # This is the number we should include on output
        ng_ix = ng_f
    else:
        params['ng'] = 0
        # This is the number we should include on output
        ng_ix = 0

    # Trivial dimensions don't have ghosts
    if params['n2'] == 1:
        ng_iy = 0
    else:
        ng_iy = ng_ix

    if params['n3'] == 1:
        ng_iz = 0
    else:
        ng_iz = ng_ix

    dx = G.dx[1]
    dy = G.dx[2]
    dz = G.dx[3]

    # Lay out the blocks and determine total mesh size
    bounds = []
    for ib in range(f.NumBlocks):
        bb = f.BlockBounds[ib]
        # Internal location of the block i.e. starting/stopping indices in the final, big mesh
        bound = [int((bb[0]+dx/2 - G.startx[1])/dx)-ng_ix, int((bb[1]+dx/2 - G.startx[1])/dx)+ng_ix,
                 int((bb[2]+dy/2 - G.startx[2])/dy)-ng_iy, int((bb[3]+dy/2 - G.startx[2])/dy)+ng_iy,
                 int((bb[4]+dz/2 - G.startx[3])/dz)-ng_iz, int((bb[5]+dz/2 - G.startx[3])/dz)+ng_iz]
        bounds.append(bound)

    # Optionally allocate enough space for ghost zones
    var = np.zeros((G.NTOT[1]+2*ng_ix, G.NTOT[2]+2*ng_iy, G.NTOT[3]+2*ng_iz))

    # Read blocks into their slices of the full mesh
    for ib in range(f.NumBlocks):
        b = bounds[ib]
        # Exclude ghost zones if we don't want them
        if ng_ix == 0 and ng_f != 0:
            if params['n2'] == 1:
                o = [ng_f, -ng_f, None, None, None, None]
            elif params['n3'] == 1:
                o = [ng_f, -ng_f, ng_f, -ng_f, None, None]
            else:
                o = [ng_f, -ng_f, ng_f, -ng_f, ng_f, -ng_f]
        else:
            o = [None, None, None, None, None, None]
        # False == don't flatten into 1D array
        fvar = f.Get(scalar_name, False)
        if fvar is None: return None
        var[b[0]+ng_ix:b[1]+ng_ix, b[2]+ng_iy:b[3]+ng_iy, b[4]+ng_iz:b[5]+ng_iz] = \
            fvar[ib,o[4]:o[5],o[2]:o[3],o[0]:o[1]].transpose(2,1,0)

    return var

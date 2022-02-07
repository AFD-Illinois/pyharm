
import numpy as np
import h5py

from .. import parameters
from ..grid import Grid
from .interface import DumpFile

# This is where we drop in the Parthenon reader
from .phdf import phdf

class KHARMAFile(DumpFile):
    """File filter for KHARMA files"""
    # Names which aren't directly prims.x or cons.x, but which we can translate
    prim_names_dict = {"RHO": "rho",
                       "UU": "u",
                       "U1": "u1",
                       "U2": "u2",
                       "U3": "u3",
                       "KTOT": "Ktot",
                       "KEL_KAWAZURA": "Kel_Kawazura",
                       "KEL_WERNER":   "Kel_Werner",
                       "KEL_ROWAN":    "Kel_Rowan",
                       "KEL_SHARMA":   "Kel_Sharma",
                       "KEL_CONSTANT": "Kel_Constant"}

    @classmethod
    def get_dump_time(cls, fname):
        """Quickly get just the simulation time represented in the dump file.
        For cutting on time without loading everything
        """
        with h5py.File(fname, 'r') as dfile:
            if 'Info' in dfile.keys():
                return dfile['Info'].attrs['Time']
        return None

    def kharma_standardize(self, var):
        """Standardize the names we're asked for, so we cache only one copy"""
        # Translate certain caps
        if var in self.prim_names_dict.keys():
            var = self.prim_names_dict[var]

        # Pick out indices and return their vectors
        ind = None
        if var[-1:] in ("1", "2", "3"):
            # Mark which index we want
            ind = int(var[-1:]) - 1
            # Then read the corresponding vector, cons/prims.u/B
            var = var[:-2] + ("B" if "B" in var[-2:] else "uvec")

        # Extend the shorthand for primitive variables to their full names in KHARMA,
        # but not other variables.
        if var in ("rho", "u", "uvec", "B"):
            var = "prims."+var
        if ("Kel" in var or "Ktot" in var) and ("cons" not in var):
            var = "prims."+var
        
        return var, ind

    def __init__(self, filename, ghost_zones=False):
        """Create an Iharm3DFile object -- note that the file handle will stay
        open as long as the object
        """
        self.fname = filename
        self.file = phdf(filename)
        self.params = self.read_params()
        self.params['ghost_zones'] = ghost_zones
        self.params['ng'] = ghost_zones * self.params['ng_file']
        self.cache = {}
        if ghost_zones and self.params['ng_file'] == 0:
            raise ValueError("Ghost zones aren't available in file {}".format(self.fname))

    # def __del__(self):
    #     # Make sure Parthenon *actually* closed the HDF5 file
    #     self.file.fid.close()

    def read_params(self):
        # TODO if reading very old output, fall back to text .par file in dumps folder
        # TODO there's likely a nice way to get all this from phdf
        if 'Input' in self.file.fid:
            par_string = self.file.fid['Input'].attrs['File']
            if type(par_string) != str:
                par_string = par_string.decode("UTF-8")
            params = parameters.parse_parthenon_dat(par_string)
        else:
            raise RuntimeError("No parameters could be found in KHARMA dump {}".format(self.fname))

        # Use Parthenon's reader for the file-specific stuff
        params['ng_file'] = self.file.NGhost * self.file.IncludesGhost
        # Set incidental parameters from what we've read
        params['t'] = self.file.Time
        params['n_step'] = self.file.NCycle
        # Add dump number if we've conformed to usual naming scheme
        fname_parts = self.fname.split("/")[-1].split(".")
        if len(fname_parts) > 2:
            try:
                params['n_dump'] = int(fname_parts[2])
            except ValueError:
                # dumps named 'final'.  Drop them?
                params['n_dump'] = fname_parts[2]

        return params

    def read_var(self, var, astype=None, slc=()):
        var, ind = self.kharma_standardize(var)
        if var in self.cache:
            if ind is not None:
                return self.cache[var][ind]
            else:
                return self.cache[var]

        # All primitives/conserved. We added this to iharm3d, special case it here
        if var == "prims":
            return np.array([self.read_var(v2) for v2 in self.file.Variables if "prims" in v2])
        elif var == "cons":
            return np.array([self.read_var(v2) for v2 in self.file.Variables if "cons" in v2])

        if var not in self.file.Variables and self.index_of(var) is None:
            raise IOError("Cannot find variable "+var+" in dump "+self.fname+". Should it have been calculated?")

        params = self.params
        # Recall ng=0 if ghost_zones is False.  Thus this says:
        # if we want ghost zones, set them in nontrivial dimensions
        ntot = (params['n1'], params['n2'], params['n3'])
        ng_ix = params['ng']
        ng_iy = params['ng'] if ntot[1] > 1 else 0
        ng_iz = params['ng'] if ntot[2] > 1 else 0
        # Even if we don't want ghosts, we'll need to cut them from the file
        ng_f = params['ng_file']
        # Finally, we need to decipher where to put each meshblock vs the whole grid
        dx = params['dx1']
        dy = params['dx2']
        dz = params['dx3']
        startx = (0., params['startx1'], params['startx2'], params['startx3'])

        # Allocate the full mesh size
        if "jcon" in var:
            out = np.zeros((4, ntot[0]+2*ng_ix, ntot[1]+2*ng_iy, ntot[2]+2*ng_iz), dtype=astype)
        elif "B" in var or "uvec" in var and ind is None:
            out = np.zeros((3, ntot[0]+2*ng_ix, ntot[1]+2*ng_iy, ntot[2]+2*ng_iz), dtype=astype)
        else:
            out = np.zeros((ntot[0]+2*ng_ix, ntot[1]+2*ng_iy, ntot[2]+2*ng_iz), dtype=astype)

        # The slice we need of each block is just ng to -ng, with 0->None for the whole slice
        o = [None if i == 0 else i for i in [ng_ix, -ng_ix, ng_iy, -ng_iy, ng_iz, -ng_iz]]

        # Lay out the blocks and determine total mesh size
        for ib in range(self.file.NumBlocks):
            bb = self.file.BlockBounds[ib]
            # Internal location of the block i.e. starting/stopping indices in the final, big mesh
            b = [int((bb[0]+dx/2 - startx[1])/dx)-ng_ix, int((bb[1]+dx/2 - startx[1])/dx)+ng_ix,
                 int((bb[2]+dy/2 - startx[2])/dy)-ng_iy, int((bb[3]+dy/2 - startx[2])/dy)+ng_iy,
                 int((bb[4]+dz/2 - startx[3])/dz)-ng_iz, int((bb[5]+dz/2 - startx[3])/dz)+ng_iz]
            # Slices with locations
            fil_slc = (ib, slice(o[4], o[5]), slice(o[2], o[3]), slice(o[0], o[1]))
            out_slc = (slice(b[0]+ng_ix, b[1]+ng_ix),
                       slice(b[2]+ng_iy, b[3]+ng_iy),
                       slice(b[4]+ng_iz, b[5]+ng_iz))
            if slc != ():
                # TODO Add this on top of the two slices
                pass

            if 'prims.rho' in self.file.Variables:
                # New file format. Read whatever
                if ind is not None: # Read index of a vector
                    # Read our optionally-sliced file into our optionally-offset memory locations
                    # False == don't flatten into 1D array
                    out[out_slc] = self.file.Get(var, False)[fil_slc + (ind,)].transpose(2,1,0)
                elif len(out.shape) == 4: # Read a full vector
                    out[(slice(None),) + out_slc] = self.file.Get(var, False)[fil_slc + (slice(None),)].transpose(3,2,1,0)
                else: # Read a scalar
                    out[out_slc] = self.file.Get(var, False)[fil_slc].transpose(2,1,0)

            else:
                # Old file formats.  If we'd split prims/B_prim:
                if "B" in var and 'c.c.bulk.B_prim' in self.file.Variables:
                        if ind is not None:
                            out[out_slc] = self.file.Get(var, False)[fil_slc + (ind,)].transpose(2,1,0)
                        else:
                            out[(slice(None),) + out_slc] = self.file.Get('c.c.bulk.B_prim', False)[fil_slc + (slice(None),)].transpose(3,2,1,0)
                else:
                    i = self.index_of(var)
                    if i is None:
                        # We're not grabbing anything except primitives from old KHARMA files.
                        raise IOError("Cannot find variable "+var+" in file "+self.file+"!")
                    elif type(i) == int:
                        out[out_slc] = self.file.Get('c.c.bulk.prims', False)[fil_slc + (i,)].transpose(2,1,0)
                    else:
                        out[Ellipsis, out_slc] = self.file.Get('c.c.bulk.prims', False)[fil_slc + (i,)].transpose(2,1,0)

        # TODO then not here
        # Currently this reads/caches a whole 
        self.cache[var] = out[slc]
        return out[slc]


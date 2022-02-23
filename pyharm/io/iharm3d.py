import h5py
import numpy as np

from .interface import DumpFile

# Treat header i/o as a part of this file,
# but don't balloon the line count
from .iharm3d_header import read_hdr, write_hdr

# This is one-and-done, no need for objects
def read_log(logfname):
    """Read an iharm3d-format log.out file into a dictionary"""
    dfile = np.loadtxt(logfname).transpose()
    # iharm3d's logs are deep magic. TODO header?
    log = {}
    log['t'] = dfile[0]
    log['rmed'] = dfile[1]
    log['pp'] = dfile[2]
    log['e'] = dfile[3]
    log['uu_rho_gam_cent'] = dfile[4]
    log['uu_cent'] = dfile[5]
    log['mdot'] = dfile[6]
    log['edot'] = dfile[7]
    log['ldot'] = dfile[8]
    log['mass'] = dfile[9]
    log['egas'] = dfile[10]
    log['Phi'] = dfile[11]
    log['phi'] = dfile[12]
    log['jet_EM_flux'] = dfile[13]
    log['divbmax'] = dfile[14]
    log['lum_eht'] = dfile[15]
    log['mdot_eh'] = dfile[16]
    log['edot_eh'] = dfile[17]
    log['ldot_eh'] = dfile[18]

    return log

# One-and-done like above.  Plus, generally user wants to select which file they're writing,
# not have a transparent interface
def write_dump(fluid_dump, fname, astype=np.float32, ghost_zones=False):
    with h5py.File(fname, "w") as outf:
        params = fluid_dump.params
        write_hdr(params, outf)

        # Per-dump single variables
        outf['t'] = params['t']
        outf['dt'] = params['dt']
        outf['dump_cadence'] = params['dump_cadence']
        outf['full_dump_cadence'] = params['dump_cadence']
        outf['is_full_dump'] = 0
        outf['n_dump'] = params['n_dump']
        outf['n_step'] = params['n_step']

        # This will fetch and write all primitive variables
        G = fluid_dump.grid
        if G.NG > 0 and not ghost_zones:
            outf["prims"] = np.einsum("p...->...p", fluid_dump['prims'][G.slices.allv + G.slices.bulk]).astype(astype)
        else:
            outf["prims"] = np.einsum("p...->...p", np.array(fluid_dump['prims'])).astype(astype)

        # Extra in-situ calculations or custom debugging additions
        if "extras" not in outf:
            outf.create_group("extras")

class Iharm3DFile(DumpFile):
    """File filter class for iharm3d dump files.
    """

    @classmethod
    def get_dump_time(cls, fname):
        """Quickly get just the simulation time represented in the dump file.
        For cutting on time without loading everything
        """
        with h5py.File(fname, 'r') as dfile:
            if 't' in dfile.keys():
                return dfile['t'][()]
            else:
                return None

    def __init__(self, filename, ghost_zones=False, params=None):
        """Create an Iharm3DFile object -- note that the file handle will stay
        open as long as the object
        """
        self.fname = filename
        self.cache = {}
        if params is None:
            self.params = self.read_params()
            self.params['ghost_zones'] = ghost_zones
            self.params['ng_file'] = self.params['ng']
            self.params['ng'] = ghost_zones * self.params['ng']
        else:
            self.params = params

    def __del__(self):
        # Try to clean up what we can. Anything that may possibly not be a simple ref
        for cache in ('cache', 'params'):
            if cache in self.__dict__:
                del self.__dict__[cache]

    def read_params(self, **kwargs):
        """Read the file header and per-dump parameters (t, dt, etc)"""
        with h5py.File(self.fname, "r") as fil:
            params = read_hdr(fil['/header'])

            # Add variables which change per-dump, recorded outside header
            for key in ['t', 'dt', 'n_step', 'n_dump', 'is_full_dump', 'dump_cadence', 'full_dump_cadence']:
                if key in fil:
                    params[key] = fil[key][()]

            # Grab the git revision if it's available, as this isn't recorded to/read from the header either
            if 'extras' in fil and 'git_version' in fil['extras']:
                params['git_version'] = fil['/extras/git_version'][()].decode('UTF-8')

            return params

    def read_var(self, var, slc=(), **kwargs):
        if var in self.cache:
            return self.cache[var]
        with h5py.File(self.fname, "r") as fil:
            # Translate the slice to a portion of the file
            # A bit overkill to stay adaptable: keeps all dimensions until squeeze in _prep_array
            # TODO ghost zones
            fil_slc = [slice(None), slice(None), slice(None)]
            if isinstance(slc, tuple) or isinstance(slc, list):
                for i in range(len(slc)):
                    if isinstance(slc[i], int) or isinstance(slc[i], np.int32) or isinstance(slc[i], np.int64):
                        fil_slc[i] = slice(slc[i], slc[i]+1)
                    else:
                        fil_slc[i] = slc[i]
            fil_slc = tuple(fil_slc)

            i = self.index_of(var)
            if i is not None:
                # This is one of the main vars in the 'prims' array
                self.cache[var] = self._prep_array(fil['/prims'][fil_slc + (i,)], **kwargs)
                return self.cache[var]
            else:
                # This is something else we should grab by name
                # Default to int type for flags
                if "flag" in var and 'astype' not in kwargs:
                    kwargs['astype'] = np.int32
                # Read desired slice
                if var in fil:
                    self.cache[var] = self._prep_array(fil[var][fil_slc], **kwargs)
                    return self.cache[var]
                elif var in fil['/extras']:
                    self.cache[var] = self._prep_array(fil['/extras/'+var][fil_slc], **kwargs)
                    return self.cache[var]
                else:
                    raise IOError("Cannot find variable "+var+" in file "+fil+"!")



    def _prep_array(self, arr, astype=None):
        """Re-order and optionally up-convert an array from a file,
        to put it in usual pyharm order/format
        """
        # Reverse indices on vectors, since most pyharm tooling expects p,i,j,k
        # See iharm_dump for analysis interface that restores i,j,k,p order
        if len(arr.shape) > 3:
            arr = np.einsum("...m->m...", arr)

        # Convert to desired type. Useful for flags.
        if astype is not None:
            arr = arr.astype(astype)
        
        return np.squeeze(arr)
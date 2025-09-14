
from enum import Enum
import numpy as np

import yt
from yt.fields.api import register_field_plugin

class FloorFlag(Enum):
    """Floor codes are non-exclusive, so it makes little sense to use a straight Enum.
    Instead, we use bitflags, starting high enough that we can stick the enum in the bottom 5 bits
    See floors.hpp in KHARMA for explanations of the flags
    """
    GEOM_RHO = 32
    GEOM_U = 64
    B_RHO = 128
    B_U = 256
    TEMP = 512
    GAMMA = 1024
    KTOT = 2048
    GEOM_RHO_FLUX = 4096
    GEOM_U_FLUX = 8192

class InversionStatus(Enum):
    """Enum denoting U to P inversion status.
    See e.g. u_to_p.hpp in KHARMA for documentation.
    """
    UNUSED = -1
    SUCCESS = 0
    NEG_INPUT = 1
    MAX_ITER = 2
    BAD_UT = 3
    BAD_GAMMA = 4
    NEG_RHO = 5
    NEG_U = 6
    NEG_RHOU = 7

class SolverStatus(Enum):
    CONVERGED = 0
    FAIL = 1
    BEYOND_TOL = 2
    BACKTRACK = 3

@register_field_plugin
def setup_kharma_fields(registry, ftype="gas", slice_info=None):
    # Alias fields: "grmhd" space should contain versions of these from any code
    for pname in ['rho', 'u', 'q', 'dP']:
        registry.alias((ftype, pname), ("parthenon", "prims."+pname))
    for name in ['divB']:
        registry.alias((ftype, name), ("parthenon", name))
    for pvname in ['uvec', 'B']:
        for i in range(3):
            registry.alias((ftype, pvname+"_%d" % (i+1)), ("parthenon", "prims."+pvname+"_%d" % i))
            registry.alias((ftype, pvname+"%d" % (i+1)), ("parthenon", "prims."+pvname+"_%d" % i))
    for fourvname in ['jcon']:
        for i in range(4):
            registry.alias((ftype, fourvname+"_%d" % i), ("parthenon", fourvname+"_%d" % i))
    for i in range(3):
        registry.alias((ftype, "U"+"%d" % (i+1)), ("parthenon", "prims.uvec_%d" % i))
    # TODO also pressure, 3vel etc
    registry.alias((ftype, "density"), (ftype, "rho"))



    # FLAGS
    # Cast the arrays
    def _fflags(field, data):
        return np.array(data["parthenon", "fflag"], dtype=np.int32)
    def _pflags(field, data):
        return np.array(data["parthenon", "pflag"], dtype=np.int32)
    def _solver_flags(field, data):
        return np.array(data["parthenon", "solve_fail"], dtype=np.int32)
    registry.add_field(
        name=("KHARMA", "fflags"),
        function=_fflags,
        sampling_type="local",
        units="",
        take_log=False,
    )
    registry.add_field(
        name=("KHARMA", "pflags"),
        function=_pflags,
        sampling_type="local",
        units="",
        take_log=False,
    )
    registry.add_field(
        name=("KHARMA", "solver_flags"),
        function=_solver_flags,
        sampling_type="local",
        units="",
        take_log=False,
    )
    
    # FLAGS
    def _floor(field, data):
        # Make sure we're &ing together ints
        return data["KHARMA", "fflags"].v & FloorFlag[field.name[1][6:]].value
    for member in FloorFlag:
        registry.add_field(
            name=("KHARMA", "FLOOR_"+member.name),
            function=_floor,
            sampling_type="local",
            units="",
            take_log=False,
        )
    
    def _pflag(field, data):
        # Make sure we're comparing ints
        return data["KHARMA", "pflags"] == InversionStatus[field.name[1][6:]].value
    for member in InversionStatus:
        registry.add_field(
            name=("KHARMA", "PFLAG_"+member.name),
            function=_pflag,
            sampling_type="local",
            units="",
            take_log=False,
        )

    def _solver_flag(field, data):
        # Make sure we're comparing ints
        return data["KHARMA", "solver_flags"] == SolverStatus[field.name[1][12:]].value
    for member in SolverStatus:
        registry.add_field(
            name=("KHARMA", "SOLVER_FLAG_"+member.name),
            function=_solver_flag,
            sampling_type="local",
            units="",
            take_log=False,
        )


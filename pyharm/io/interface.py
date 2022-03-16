
"""This interface provides the list of functions to implement for new file filters for pyharm.
A good base set of variables would be to provide something logical for the strings RHO:B3 in both
lists below.  The provided index_of() function returns the expected indices 1-8 for these variables --
if your file has an array prims[] you can likely return prims[:,:,:,index_of(var)] and cover many/most calls.

Constructor signature:
__init__(self, ghost_zones=False)

Where 

The constructor must initialize a dictionary member self.params, containing at least enough members to initialize
a Grid object (see Grid constructor docstring), as well as any single-scalar properties that analysis or plotting
code will need to access (e.g. fluid gamma, BH spin, times & timesteps, etc).
Usually this is done via a member function self.read_params, which may be called on its own in future

"""

class DumpFile(object):
    """Interface providing consistent methods to read and write dump files, or some slice thereof.
    Subclasses keep a persistent file handle until they are destroyed, and read on-demand.
    """

    prim_names_iharm = ("RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3",
                        "KTOT", "KEL_CONSTANT", "KEL_KAWAZURA", "KEL_WERNER", "KEL_ROWAN", "KEL_SHARMA")
    
    @classmethod
    def index_of(cls, vname, eprim_names=None, eprim_indices=None):
        # This is provided in the interface, as anything outputting just an array (iharmXd, KORAL, etc)
        # uses this ordering of variables.  Other formats can just avoid calling index_of
        # Custom names, e.g. e-
        if eprim_names is not None and vname.upper() in eprim_names:
            return eprim_indices[eprim_names.index(vname.upper())]
        # Normal names
        elif vname.upper() in cls.prim_names_iharm:
            return cls.prim_names_iharm.index(vname)
        # Vectors
        elif vname == "uvec":
            return slice(cls.index_of("u1"), cls.index_of("u3")+1)
        elif vname == "B":
            return slice(cls.index_of("B1"), cls.index_of("B3")+1)
        # All
        elif vname == "prims" or vname == "primitives" or vname == "all":
            return slice()
        else:
            return None

    @classmethod
    def get_dump_time(cls, fname):
        raise NotImplementedError

    def read_var(self, var, slice=None):
        """Read a variable 'var' from the file as a numpy array.
        Optionally read ghost zones if specified, or read only a slice 'slice' of the values from the file.
        Returns 'None' if the variable doesn't exist in the file.
        """
        raise NotImplementedError

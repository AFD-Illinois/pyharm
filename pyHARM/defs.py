# Definitions of enums and slices used throughout the code

from enum import Enum


class Loci(Enum):
    """Location enumerated value.
    Locations are defined by:
    ^ theta
    |----------------------
    |                     |
    |                     |
    |FACE1   CENT         |
    |                     |
    |CORN    FACE2        |
    -------------------------> R
    With FACE3 as the plane in phi"""
    FACE1 = 0
    FACE2 = 1
    FACE3 = 2
    CENT = 3
    CORN = 4


class Var(Enum):
    """All variables HARM currently supports evolving.
    May not all be used in a given run.
    See Gammie et al '03, Gammie & McKinney '04, Ressler et al '17 for definitions
    """
    RHO = 0
    UU = 1
    U1 = 2
    U2 = 3
    U3 = 4
    B1 = 5
    B2 = 6
    B3 = 7
    KTOT = 8
    KEL = 9


class Slices:
    """Name commonly used slices of the full grid :
    The "bulk" fluid is the physical zones only, without any of the surrounding "ghost" zones obtained
    from other MPI ranks.  When parts of the set of ghost zones are used, they're called a "halo"
    Primitive physical values are also given slices, to be able to add them to zone slices to index P
    """

    def __init__(self, ng):
        # Slices to represent variables, to add to below for picking out e.g. bulk of RHO
        self.allv = (slice(None),)
        self.RHO = (Var.RHO.value,)
        self.UU = (Var.UU.value,)
        self.U1 = (Var.U1.value,)
        self.U2 = (Var.U2.value,)
        self.U3 = (Var.U3.value,)
        self.U3VEC = (slice(Var.U1.value, Var.U3.value + 1),)
        self.B1 = (Var.B1.value,)
        self.B2 = (Var.B2.value,)
        self.B3 = (Var.B3.value,)
        self.B3VEC = (slice(Var.B1.value, Var.B3.value+1),)
        self.KTOT = (Var.KTOT.value,)
        self.KEL = (Var.KEL.value,)

        # General slice for 3vec portion of 4vec
        self.VEC3 = (slice(1, None),)

        # Single slices for putting together operations in bounds.py.  May be replaced by loopy kernels
        # Name single slices for character count
        self.a = slice(None)
        self.b = slice(ng, -ng)
        self.r1 = slice(ng + 1, -ng + 1)
        self.l1 = slice(ng - 1, -ng - 1)

        self.bulk = (self.b, self.b, self.b)
        self.all = (self.a, self.a, self.a)
        # "Halo" of 1 zone
        self.bh1 = slice(ng - 1, -ng + 1)
        self.bulkh1 = (self.bh1, self.bh1, self.bh1)
        # Right halo only
        self.brh1 = slice(ng, -ng + 1)
        self.bulkrh1 = (self.brh1, self.brh1, self.brh1)

        # Slices to produce bulk-sized output from finite difference kernels
        self.diffr1 = (slice(ng + 1, -ng + 1), self.b, self.b)
        self.diffr2 = (self.b, slice(ng + 1, -ng + 1), self.b)
        self.diffr3 = (self.b, self.b, slice(ng + 1, -ng + 1))

        # Name boundaries slices for readability
        # Left side
        self.ghostl = slice(0, ng)
        self.boundl = slice(ng, 2*ng)
        self.boundl_r = slice(2 * ng, ng, -1)  # Reverse
        self.boundl_o = slice(ng, ng + 1)  # Outflow (1-zone slice for replication)
        # Right side
        self.ghostr = slice(-ng, None)
        self.boundr = slice(-2 * ng, -ng)
        self.boundr_r = slice(-ng, -2 * ng, -1)
        self.boundr_o = slice(-ng - 1, -ng)

    def geom_slc(self, slc):
        """Return the version of a 3D slice suitable for 2D geometry variables"""
        return slc[:2] + (None,)


class Shapes:
    """Shape of the geometry, and the grid with & without ghost zones.
    Adheres to the same "bulk" naming as above for versions excluding ghost zones.
    """
    def __init__(self, G, params):
        # Shapes for allocation
        self.geom_scalar = (G.GN[1], G.GN[2])
        self.geom_vector = (4,) + self.geom_scalar
        # Admittedly this is not really the meaning of tensor.
        # tensor2d is needless specification though
        self.geom_tensor = (4,) + self.geom_vector

        self.locus_geom_scalar = (len(list(Loci)),) + self.geom_scalar
        self.locus_geom_vector = (len(list(Loci)),) + self.geom_vector
        self.locus_geom_tensor = (len(list(Loci)),) + self.geom_tensor

        self.grid_scalar = (G.GN[1], G.GN[2], G.GN[3])
        self.grid_vector = (4,) + self.grid_scalar
        self.grid_tensor = (4,) + self.grid_vector
        self.grid_primitives = (params['n_prims'],) + self.grid_scalar

        self.bulk_scalar = (G.N[1], G.N[2], G.N[3])
        self.bulk_vector = (4,) + self.bulk_scalar
        self.bulk_tensor = (4,) + self.bulk_vector
        self.bulk_primitives = (params['n_prims'],) + self.bulk_scalar

        self.halo1_scalar = (G.N[1] + 2, G.N[2] + 2, G.N[3] + 2)
        self.halo1_primitives = (params['n_prims'],) + self.halo1_scalar


        # ISL language definitions of common kernel sizes
        # TODO any advantage to actually _defining_ the numbers?
        self.isl_geom_scalar = """{ [i,j]: 0 <= i < n1 and 0 <= j < n2 }"""
        self.isl_geom_vector = """{ [mu,nu,i,j]: 0 <= mu < ndim and 0 <= i < n1 and 0 <= j < n2 }"""
        self.isl_geom_tensor = """{ [mu,nu,i,j]: 0 <= mu < ndim and 0 <= nu < ndim and 0 <= i < n1 and 0 <= j < n2 }"""
        self.isl_geom_primitives = """{ [p,i,j,k]: 0 <= p < nprims and 0 <= i < n1 and 0 <= j < n2 and 0 <= k < n3 }"""

        self.isl_grid_scalar = """{ [i,j,k]: 0 <= i < n1 and 0 <= j < n2 and 0 <= k < n3 }"""
        self.isl_grid_vector = """{ [mu,i,j,k]: 0 <= mu < ndim and 0 <= i < n1 and 0 <= j < n2 and 0 <= k < n3 }"""
        self.isl_grid_tensor = """{ [mu,nu,i,j,k]: 0 <= mu < ndim and 0 <= nu < ndim and
                                        0 <= i < n1 and 0 <= j < n2 and 0 <= k < n3 }"""
        self.isl_grid_3vector = """{ [mu,i,j,k]: 0 <= mu < ndim-1 and 0 <= i < n1 and 0 <= j < n2 and 0 <= k < n3 }"""
        self.isl_grid_3tensor = """{ [mu,nu,i,j,k]: 0 <= mu < ndim-1 and 0 <= nu < ndim-1 and
                                        0 <= i < n1 and 0 <= j < n2 and 0 <= k < n3 }"""

        self.isl_grid_primitives = """{ [p,i,j,k]: 0 <= p < nprim and 0 <= i < n1 and 0 <= j < n2 and 0 <= k < n3 }"""
        self.isl_grid_primitives_fixup = """{ [p,i,j,k,l,m,n]: 0 <= p < nprim
                                    and 0 <= i < n1 and 0 <= j < n2 and 0 <= k < n3 
                                    and -1 <= l <= 1 and -1 <= m <= 1 and -1 <= n <= 1}"""
        self.isl_grid_scalar_fixup = """{ [p,i,j,k,l,m,n]: 0 <= i < n1 and 0 <= j < n2 and 0 <= k < n3 
                                    and -1 <= l <= 1 and -1 <= m <= 1 and -1 <= n <= 1}"""

        self.assume_grid = "n1 mod 2 = 0 and n2 mod 2 = 0 and n3 mod 2 = 0 and n1 >= 1 and n2 >= 1 and n3 >= 1 "
        self.assume_grid_primitives = self.assume_grid + "and nprim >= 1 "


__license__ = """
 File: defs.py
 
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

from enum import Enum
import numpy as np

__doc__ = \
"""Definitions of enums used throughout pyharm, and common slices
to make vectorized operations a bit more readable.

Some of this is pretty old.
"""

class Loci(Enum):
    """Location enumerated value.
    Locations are defined by::

        ^ theta
        |----------------------
        |                     |
        |                     |
        |FACE1     CENT       |
        |                     |
        |CORN      FACE2      |
        -------------------------> R
    
    And FACE3 as the plane in phi.
    This enum corresponds to the various request-able grid locations:
    center is the default, with the others generally called
    "corner", "face1", "face2", "face3"
    """
    FACE1 = 0
    FACE2 = 1
    FACE3 = 2
    CENT = 3
    CORN = 4

class InversionStatus(Enum):
    """Enum denoting U to P inversion status.
    See e.g. u_to_p.hpp in KHARMA for documentation.
    """
    unused = -1
    success = 0
    neg_input = 1
    max_iter = 2
    bad_ut = 3
    bad_gamma = 4
    neg_rho = 5
    neg_u = 6
    neg_rhou = 7

class FloorFlag_iharm3d(Enum):
    """Bit masks for HARM floor codes."""
    HIT_FLOOR_GEOM_RHO = 1
    HIT_FLOOR_GEOM_U = 2
    HIT_FLOOR_B_RHO = 4
    HIT_FLOOR_B_U = 8
    HIT_FLOOR_TEMP = 16
    HIT_FLOOR_GAMMA = 32
    HIT_FLOOR_KTOT = 64
    FLOOR_UTOP_FAIL = 128

class FloorFlag_KHARMA(Enum):
    """Floor codes are non-exclusive, so it makes little sense to use a straight Enum.
    Instead, we use bitflags, starting high enough that we can stick the enum in the bottom 5 bits
    See floors.hpp in KHARMA for explanations of the flags
    """
    HIT_FLOOR_GEOM_RHO = 32
    HIT_FLOOR_GEOM_U = 64
    HIT_FLOOR_B_RHO = 128
    HIT_FLOOR_B_U = 256
    HIT_FLOOR_TEMP = 512
    HIT_FLOOR_GAMMA = 1024
    HIT_FLOOR_KTOT = 2048
    #  Separate flags for when the floors are applied after reconstruction.
    #  Not yet used, as this will likely have some speed penalty paid even if
    #  the flags aren't written
    HIT_FLOOR_GEOM_RHO_FLUX = 4096
    HIT_FLOOR_GEOM_U_FLUX = 8192

class Slices:
    """Object to hold a bunch of named slices.
    The "bulk" fluid is the physical zones only, without any of the surrounding "ghost" zones obtained
    from other MPI ranks.
    When parts of the set of ghost zones are used, they're called a "halo"
    """

    def __init__(self, ng):
        # General slice for all of a given dimension
        self.allv = (slice(None),)

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

        # Boundaries
        self.ghostl = slice(0, ng)
        self.ghostr = slice(-ng, None)

    def geom_slc(self, slc):
        """Return the version of a 3D slice suitable for 2D geometry variables"""
        if len(slc) > 2:
            if isinstance(slc[2], int) or isinstance(slc[2], np.int32) or isinstance(slc[2], np.int64):
                return slc[:2] + (0,)
            else:
                return slc[:2] + (slice(None),)
        else:
            return slc


class Shapes:
    """Shape of the geometry, and the grid with & without ghost zones.
    Adheres to the same "bulk" naming as above for versions excluding ghost zones.
    """
    def __init__(self, G, params):
        # Shapes for allocation
        self.geom_scalar = (G.GN[1], G.GN[2], 1)
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

        self.bulk_scalar = (G.N[1], G.N[2], G.N[3])
        self.bulk_vector = (4,) + self.bulk_scalar
        self.bulk_tensor = (4,) + self.bulk_vector

        self.halo1_scalar = (G.N[1] + 2, G.N[2] + 2, G.N[3] + 2)
__license__ = """
 File: grid.py
 
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

from pyharm.defs import Loci, Slices, Shapes
from pyharm.coordinates import *


def make_some_grid(system, n1=128, n2=128, n3=128, a=0, hslope=0.3,
                    r_in=None, r_out=1000, caches=True, cache_conn=False):
    """Convenience function for generating grids with particular known parameters.

    :param system: coordinate system, denoted 'eks', 'mks', 'fmks', 'minkowski', or
                 anything exotic defined in :mode`pyharm.coordinates`

    All other parameters are as described in Grid.__init__, and are optional with
    Illinois-centric defaults. If not given, 'r_in' will be chosen to put 5 zones
    inside the event horizon as done in iharm3d/KHARMA.
    """

    params = {}
    params['coordinates'] = system
    params['n1tot'] = n1
    params['n2tot'] = n2
    params['n3tot'] = n3
    params['n1'] = n1
    params['n2'] = n2
    params['n3'] = n3
    params['ng'] = 0

    if system == 'minkowski' or system == 'cartesian':
        params['x1min'] = 0
        params['x2min'] = 0
        params['x3min'] = 0
        params['x1max'] = 1
        params['x2max'] = 1
        params['x3max'] = 1
    elif 'ks' in system:
        params['a'] = a
        params['r_out'] = r_out
        if r_in is not None:
            params['r_in'] = r_in
        if 'mks' in system:
            params['hslope'] = hslope
        if system == 'fmks' or system == 'mmks':
            params['poly_xt'] = 0.82
            params['poly_alpha'] = 14.0
            params['mks_smooth'] = 0.5

    return Grid(params, caches=caches, cache_conn=cache_conn)



class Grid:
    """The Grid object divides a domain in native coordinates into zones, and caches the
    local metric (and some other convenient information) at several locations in each zone.
    The object can be used to consult the grid size/shape for global calculations, and raise and
    lower the indices of fluid 4-vectors.
    """

    def __init__(self, params, caches=True, cache_conn=False):
        """
        Initialize a Grid object.  Note that "params" is usually filled by reading a file, not manually:
        for manual Grid creation, see :func:`pyharm.grid.make_some_grid`.

        :param caches: Whether to cache gcon/gcov/gdet at zone centers/faces. Usually desired.
        :param cache_conn: Whether to cache connection coefficients (all 64/zone) at zone centers. Usually not desired.

        :param params:
            | Dictionary containing the following parameters, depending on coordinate system:
            | n{1,2,3}tot: total number of physical grid zones in each direction
            | ng: number of "ghost" zones in each direction to add for boundary conditions/domain decomposition
            | coordinates: name of the coordinate system (value in parens below)
            |
            | === For Minkowski coordinates ("minkowski"): ===
            | x{1,2,3}min: location of nearest corner of first grid zone [0,0,0]
            | x{1,2,3}max: location of farthest corner of last grid zone [n1tot, n2tot, n3tot]
            |
            | === For Spherical Kerr-Schild coordinates ("ks"): ===
            | a: black hole spin
            | r_out: desired location of the outer edge of the farthest radial zone on the grid
            |
            | === For Modified Kerr-Schild coordinates ("mks"): ===
            | hslope: narrowing parameter for X2, defined in `Gammie et. al. (2003) <https://doi.org/10.1086/374594>`_
            |
            | === For Modified Modified Kerr-Schild coordinates ("mmks"): ===
            | All of the MKS parameters, plus
            | poly_xt, poly_alpha: See the AFD `docs wiki <https://github.com/AFD-Illinois/docs/wiki/Coordinates>`_
            |
            | === For "Funky" Modified Kerr-Schild coordinates ("fmks"): ===
            | All of the MMKS parameters, plus
            | mks_smooth: See HARM docs wiki
        """
        # Set the basic grid parameters
        # Total grid size (all MPI processes)
        self.params = params
        self.cache = {}
        if 'n1tot' in params:
            self.NTOT = np.array([1, params['n1tot'], params['n2tot'], params['n3tot']])
        else:
            self.NTOT = np.array([1, params['n1'], params['n2'], params['n3']])
        # Indices of first & last zone in global grid
        self.global_start = np.array([0, 0, 0])  # TODO someday this and N need modification
        self.global_stop = self.NTOT[1:]
        # Size of this grid
        self.N = self.NTOT  # TODO do I really want to carry around a useless first index?
        # Number of ghost zones
        if 'ng' in params:
            self.NG = params['ng']
        else:
            self.NG = 0
        # Size of grid in-memory (with ghost zones where appropriate)
        self.GN = self.N + (self.N > 1) * 2*self.NG

        # Slices and shapes sub-objects: to hold numbers in convenient namespaces
        self.slices = Slices(self.NG)
        self.shapes = Shapes(self, params)

        if params['coordinates'] in ["minkowski", "cartesian"]:
            # There are no parameters to Minkowski coordinates, so this is a class, not an object
            self.coords = Minkowski
        # MKS, FMKS expect the parameters a, hslope
        elif params['coordinates'] == "mmks" or params['coordinates'] == "fmks":
            # FMKS additionally requires poly_xt, poly_alpha, mks_smooth
            self.coords = FMKS(params)
        elif params['coordinates'] == "cmks":
            # MMKS additionally requires poly_xt and poly_alpha
            self.coords = CMKS(params)
        elif params['coordinates'] == "bhac_mks":
            # BHAC's MKS
            self.coords = BHAC_MKS(params)
        elif params['coordinates'] == "mks3" or params['coordinates'] == "mks2":
            # KORAL's MKS.  MKS2 == MKS3 w/ MY1,2,MP0 == 0
            self.coords = MKS3(params)
        elif params['coordinates'] == "mks":
            self.coords = MKS(params)
        elif params['coordinates'] == "eks":
            self.coords = EKS(params)
        elif params['coordinates'] == "superexp":
            self.coords = SEKS(params)
        elif params['coordinates'] == "ks":
            self.coords = KS(params)
        elif params['coordinates'] == "bl":
            self.coords = BL(params)
        else:
            raise ValueError("metric is {}!! must be minkowski, mks, mmks, or fmks".format(params['coordinates']))

        # If we got native coordinates, use those
        if 'x1min' in params:
            self.startx = np.array([0, params['x1min'], params['x2min'], params['x3min']])
        elif 'startx1' in params:
            self.startx = np.array([0, params['startx1'], params['startx2'], params['startx3']])
        else:
            # Ask our new coordinate system where to start/stop the native grid,
            # so it aligns with the KS boundaries we've been assigned
            self.startx = self.coords.native_startx(params)
            if params['coordinates'] not in ["minkowski", "cartesian"] and self.startx[1] < 0.0:
                raise ValueError("Not enough radial zones! Increase N1!")

        if 'dx1' in params:
            self.dx = np.array([0, params['dx1'], params['dx2'], params['dx3']])
            self.stopx = self.startx + self.NTOT * self.dx
        else:
            self.stopx = self.coords.native_stopx(params)
            self.dx = (self.stopx - self.startx) / self.NTOT

        self.dV = self.dx[1]*self.dx[2]*self.dx[3]

        # If we're in Cartesian Minkowski, keep a "grid" of just one zone
        if caches and (self.coords == Minkowski):
            # Shapes. Store like a 0-dim array:
            # locations, tensor dims, grid dims
            self.gcov = np.zeros((5,4,4,1,1,1))
            self.gdet = np.zeros((5,1,1,1))
            # Replicate
            self.gcon = np.zeros_like(self.gcov)
            self.lapse = np.zeros_like(self.gdet)

            for loc in Loci:
                ilist = np.arange(1)
                jlist = np.arange(1)
                x = self.coord(ilist, jlist, [0,], loc)

                gcov_loc = self.coords.gcov(x)
                gcon_loc = self.coords.gcon_from_gcov(gcov_loc)
                gdet_loc = self.coords.gdet_from_gcov(gcov_loc)

                self.gcov[loc.value] = gcov_loc
                self.gcon[loc.value] = gcon_loc
                self.gdet[loc.value] = gdet_loc
                self.lapse[loc.value] = 1./np.sqrt(-gcon_loc[0, 0])

        elif caches:
            # Declare shapes so we can fill locations one by one
            self.gcov = np.zeros(self.shapes.locus_geom_tensor)
            self.gdet = np.zeros(self.shapes.locus_geom_scalar)
            # Replicate
            self.gcon = np.zeros_like(self.gcov)
            self.lapse = np.zeros_like(self.gdet)

            for loc in Loci:
                ilist = np.arange(self.GN[1])
                jlist = np.arange(self.GN[2])
                x = self.coord(ilist, jlist, [0,], loc)

                # Save zone centers to calculate connection coefficients
                if cache_conn and loc == Loci.CENT:
                    self.conn = self.coords.conn_func(x)

                self.gcov[loc.value] = self.coords.gcov(x)
                self.gcon[loc.value] = self.coords.gcon_from_gcov(self.gcov[loc.value])
                self.gdet[loc.value] = self.coords.gdet_from_gcov(self.gcov[loc.value])
                self.lapse[loc.value] = 1./np.sqrt(-self.gcon[loc.value, 0, 0])

    def __del__(self):
        # Try to clean up what we can. Anything that may possibly not be a simple ref
        for cache in ('gcon', 'gcov', 'gdet', 'lapse', 'conn', 'slices', 'shapes', 'coords', 'params', 'cache'):
            if cache in self.__dict__:
                del self.__dict__[cache]

    ### COORDINATES
    def coord(self, i, j, k, loc=Loci.CENT, squeeze=False):
        """Get the position x of zone(s) i,j,k, in _native_ coordinates

        If given lists of i,j,k, this returns x[NDIM,len(i),len(j),len(k)] via np.meshgrid().
        Any index given as a single value is suppressed on output, down to x[NDIM]

        All functions in coordinates.py which take coordinates "x" also accept a grid of the form this returns.
        """
        i += self.global_start[0]
        j += self.global_start[1]
        k += self.global_start[2]

        # Create list of 3 lists of coordinates
        x = [np.zeros(1)]
        for index in i, j, k:
            if isinstance(index, np.ndarray) and len(index.shape) == 1:
                x.append(np.zeros_like(index))
            else:
                x.append(np.zeros(1))

        for mu in range(4):
            x[mu] = self.startx[mu]

        NG = self.NG
        dx = self.dx
        if loc == Loci.FACE1:
            x[1] += (i - NG) * dx[1]
            x[2] += (j + 0.5 - NG) * dx[2]
            x[3] += (k + 0.5 - NG) * dx[3]
        elif loc == Loci.FACE2:
            x[1] += (i + 0.5 - NG) * dx[1]
            x[2] += (j - NG) * dx[2]
            x[3] += (k + 0.5 - NG) * dx[3]
        elif loc == Loci.FACE3:
            x[1] += (i + 0.5 - NG) * dx[1]
            x[2] += (j + 0.5 - NG) * dx[2]
            x[3] += (k - NG) * dx[3]
        elif loc == Loci.CENT:
            x[1] += (i + 0.5 - NG) * dx[1]
            x[2] += (j + 0.5 - NG) * dx[2]
            x[3] += (k + 0.5 - NG) * dx[3]
        elif loc == Loci.CORN:
            x[1] += (i - NG) * dx[1]
            x[2] += (j - NG) * dx[2]
            x[3] += (k - NG) * dx[3]
        else:
            raise ValueError("Invalid coordinate location!")

        return np.array(np.meshgrid(x[0], x[1], x[2], x[3]))[Ellipsis, 0, :, :]

    def coord_bulk(self, loc=Loci.CENT, mesh=False):
        """Return a 3D array of all position vectors X within the physical zones.
        See coord() for use.
        """
        if mesh:
            return self.coord(np.arange(self.N[1]+1)+self.NG,
                    np.arange(self.N[2]+1)+self.NG,
                    np.arange(self.N[3]+1)+self.NG, loc=Loci.CORN)
        else:
            return self.coord(np.arange(self.N[1])+self.NG,
                            np.arange(self.N[2])+self.NG,
                            np.arange(self.N[3])+self.NG, loc=loc)

    def coord_all(self, loc=Loci.CENT, mesh=False):
        """Like coord_bulk, but including ghost zones"""
        if mesh:
            return self.coord(np.arange(self.GN[1]+1),
                            np.arange(self.GN[2]+1),
                            np.arange(self.GN[3]+1), loc=Loci.CORN)
        else:
            return self.coord(np.arange(self.GN[1]),
                            np.arange(self.GN[2]),
                            np.arange(self.GN[3]), loc=loc)

    def coord_ij(self, at=0):
        """Get just a 2D meshgrid of locations, usually for plotting"""
        return self.coord(np.arange(self.GN[1]), np.arange(self.GN[2]), at, loc=Loci.CENT)

    def coord_ik(self, at=0):
        """Get just a 2D meshgrid of locations, usually for plotting"""
        return self.coord(np.arange(self.GN[1]), at, np.arange(self.GN[3]), loc=Loci.CENT)

    def coord_jk(self, at=0):
        """Get just a 2D meshgrid of locations, usually for plotting"""
        return self.coord(at, np.arange(self.GN[2]), np.arange(self.GN[3]), loc=Loci.CENT)

    def coord_ij_mesh(self, at=0):
        """Get just a 2D meshgrid of locations, usually for plotting"""
        return self.coord(np.arange(self.GN[1]+1), np.arange(self.GN[2]+1), at, loc=Loci.CORN)

    def coord_ik_mesh(self, at=0):
        """Get just a 2D meshgrid of locations, usually for plotting"""
        return self.coord(np.arange(self.GN[1]+1), at, np.arange(self.GN[3]+1), loc=Loci.CORN)

    def coord_jk_mesh(self, at=0):
        """Get just a 2D meshgrid of locations, usually for plotting"""
        return self.coord(at, np.arange(self.GN[2]+1), np.arange(self.GN[3]+1), loc=Loci.CORN)

    ### OPERATIONS
    def lower_grid(self, vcon, loc=Loci.CENT):
        """Lower a grid of contravariant rank-1 tensors to covariant ones."""
        return np.einsum("ij...,j...->i...", self.gcov[loc.value], vcon)

    def raise_grid(self, vcov, loc=Loci.CENT):
        """Raise a grid of covariant rank-1 tensors to contravariant ones."""
        return np.einsum("ij...,j...->i...", self.gcon[loc.value], vcov)

    def dot(self, ucon, ucov):
        """Inner product along first index."""
        return np.einsum("i...,i...", ucon, ucov)

    def dt_light(self):
        """Returns the light crossing time of the smallest zone in the grid"""
        # Following stolen from bhlight's dt_light calculation

        dt_light_local = np.zeros((self.N[1], self.N[2]), dtype=np.float64)
        gcon = self.gcon
        CENT = Loci.CENT.value
        for mu in range(1,4):
            cplus = np.abs((-gcon[CENT, 0, mu] +
                            np.sqrt(gcon[CENT, 0, mu]**2 -
                                    gcon[CENT, mu, mu] * gcon[CENT, 0, 0])) /
                           gcon[CENT, 0, 0])

            cminus = np.abs((-gcon[CENT, 0, mu] -
                             np.sqrt(gcon[CENT, 0, mu]**2 -
                                     gcon[CENT, mu, mu] * gcon[CENT, 0, 0])) /
                            gcon[CENT, 0, 0])
            light_phase_speed = np.maximum.reduce([cplus, cminus])

            light_phase_speed = np.where(gcon[CENT, 0, mu] ** 2 -
                                         gcon[CENT, mu, mu] * gcon[CENT, 0, 0] >= 0.,
                                         light_phase_speed, 1e-20)

            dt_light_local += 1. / (self.dx[mu] / light_phase_speed)

        dt_light_local = 1. / dt_light_local

        return np.min(dt_light_local)

    ### PLOTTING/CONVENIENCE
    def get_xz_locations(self, mesh=False, native=False, half_cut=False, log_r=False):
        """Get the mesh locations x_ij and z_ij needed for plotting a poloidal slice.
        By default, gets locations at zone centers in slices phi=0,180.
        Note there is no need for an 'at' parameter, at least for plotting: 2D plots should be face-on.

        :param mesh: get mesh corners rather than centers, for flat shading
        :param native: get native X1/X2 coordinates rather than Cartesian x,z locations
        :param half_cut: get only the slice at phi=0
        """
        # TODO cache this!
        # TODO oblate option for x=sqrt(r^2 + a^2) rather than r
        if native:
            # We always want one "pane" when plotting in native coords
            half_cut = True
        if mesh:
            # We need a continouous set of corners representing phi=0/pi
            m = self.coord_ij_mesh(at=(0, self.NTOT[3]//2))
            if half_cut:
                m = m[Ellipsis, 0]
            else:
                # Append reversed in th.  We're now contiguous over th=180, so we remove the last
                # (or after reversal, first) zone of the flipped (left) side
                m = np.append(m[:, :, :, 0], np.flip(m[:, :, :-1, 1], 2), 2)
        else:
            # Version for zone centers doesn't need the extra 
            m = self.coord_ij(at=(0, self.NTOT[3]//2))
            if half_cut:
                m = m[Ellipsis, 0]
            else:
                m = np.append(m[Ellipsis, 0], np.flip(m[Ellipsis, 1], 2), 2)
        if native:
            x = m[1]
            z = m[2]
        else:
            x = self.coords.cart_x(m, log_r)
            z = self.coords.cart_z(m, log_r)

        return np.squeeze(x), np.squeeze(z)

    def get_xy_locations(self, mesh=False, native=False, log_r=False):
        """Get the mesh locations x_ij and y_ij needed for plotting a midplane slice.
        Note there is no need for an 'at' parameter, at least for plotting: 2D plots should be face-on.

        :param mesh: get mesh corners rather than centers, for flat shading
        :param native: get native X1/X3 coordinates rather than Cartesian x,z locations
        :param log_r: logarithmically compress the radial coordinate
        """
        # TODO cache this!
        # TODO oblate option for x,y=sqrt(r^2 + a^2) rather than r
        if mesh:
            m = self.coord_ik_mesh(at=self.NTOT[2]//2)
        else:
            m = self.coord_ik(at=self.NTOT[2]//2)

        if native:
            x = m[1]
            y = m[3]
        else:
            x = self.coords.cart_x(m, log_r)
            y = self.coords.cart_y(m, log_r)
        
        return np.squeeze(x), np.squeeze(y)

    def get_xz_areas(self, **kwargs):
        """Get cell areas in the plotting plane using the trapezoid area function from cell corners"""
        x, z = self.get_xz_locations(mesh=True, **kwargs)
        x1 = x[:-1,:-1]; z1 = z[:-1,:-1]
        x2 = x[1: ,:-1]; z2 = z[1: ,:-1]
        x3 = x[1: ,1: ]; z3 = z[1: ,1: ]
        x4 = x[:-1,1: ]; z4 = z[:-1,1: ]
        return 0.5 * np.abs(x1*z2+x2*z3+x3*z4+x4*z1 - x2*z1-x3*z2-x4*z3-x1*z4)

    def get_thphi_locations(self, at, mesh=False, native=False, bottom=False, projection='mercator'):
        """Get the mesh locations x_ij and y_ij needed for plotting a th-phi slice.
        This can be done in a bunch of ways controlled with options

        :param mesh: get mesh corners rather than centers, for flat shading
        :param native: get native X1/X3 coordinates rather than Cartesian x,z locations
        :param bottom: take the view from -z axis instead of +z axis
        :param projection:
            | "mercator": default, project theta on Y-axis and phi on X-axis. Differs from 'native' due to midplane compression.
            | "polar": view down from +z.  Or with 'bottom', view up from -Z.
            | "flattened_polar": reinterpret as polar coordinates, theta -> r, phi -> phi
        """
        # TODO cache this!
        j_slice = slice(None)
        if projection in ('polar', 'flattened_polar'):
            if bottom:
                j_slice = slice(self.NTOT[2]//2, None)
            else:
                j_slice = slice(None, self.NTOT[2]//2)

        if mesh:
            m = self.coord_jk_mesh(at=at)[j_slice]
        else:
            m = self.coord_jk(at=at)[j_slice]

        if native:
            # This puts phi on the x-axis,
            # which is much more understandable for movies
            x = m[3]
            y = m[2]
        elif projection == 'mercator':
            x = self.coords.phi(m)
            y = self.coords.th(m)
        elif projection == 'polar':
            x = self.coords.cart_x(m)
            y = self.coords.cart_y(m)
        elif projection == 'flattened_polar':
            x = self.coords.th(m) * np.cos(self.coords.phi(m))
            y = self.coords.th(m) * np.sin(self.coords.phi(m))
        
        return np.squeeze(x), np.squeeze(y)

    def __contains__(self, key):
        """Whether the given key would return something from this object.
        Generally for dictionaries the syntax is "if x in dict.keys()" or "if x in dict" but we can't enumerate possibilities.
        So instead, check strings on the fly.
        """
        if isinstance(key, int):
            return False
        if key in self.__dict__:
            return True
        elif key in self.cache:
            return True
        elif key[:7] == 'pcoord_':
            return key[8:] in self
        elif key in ('n1', 'n2', 'n3', 'r', 'th', 'phi', 'r1d', 'th1d', 'phi1d', 'x', 'y', 'z', 'X1', 'X2', 'X3',
                     'dXdx', 'dxdX', 'dxdX_cart', 'dXdx_cart', 'dxdX_bl', 'dXdx_bl', 'gcon_ks', 'gcov_ks'):
            return True
        else:
            return False

    def __getitem__(self, key):
        """This function works something like its companion in FluidDump:
        It parses a dictionary member "request" and returns various members based on it.
        This function also allows slicing -- slices must be 3D like for fluid dumps, though
        only the X1 and X2 slices are applied.
        """
        if type(key) in (int,):
            return self
        if type(key) in (list, tuple) and type(key[0]) in (int, np.int32, np.int64, slice):
            # Grids also support slicing, see analogue in FluidDump
            slc = self.slices.geom_slc(key) # cut 3rd index, geometry is 2D
            relevant_0 = isinstance(slc[0], int) or isinstance(slc[0], np.int32) or isinstance(slc[0], np.int64) \
                         or isinstance(slc[0].start, int) or isinstance(slc[0].stop, int)
            relevant_1 = len(slc) > 1 and isinstance(slc[1], int) or isinstance(slc[1], np.int32) or isinstance(slc[1], np.int64) \
                         or isinstance(slc[1].start, int) or isinstance(slc[1].stop, int)
            relevant_2 = len(slc) > 2 and isinstance(slc[2], int) or isinstance(slc[2], np.int32) or isinstance(slc[2], np.int64) \
                         or isinstance(slc[2].start, int) or isinstance(slc[2].stop, int)
            if not (relevant_0 or relevant_1 or relevant_2):
                return self
            # Otherwise it's worth it to make a new grid & return a part.
            # TODO this can be a proper copy constructor right?
            #print("Grid slice copy, ",key)
            out = Grid(self.params, caches=False)
            #out = copy.deepcopy(self) # In case this proves faster
            out.slice = slc
            # Revise size numbers for this grid
            for i in range(len(out.slice)):
                if isinstance(out.slice[i], int) or isinstance(slc[i], np.int32) or isinstance(slc[i], np.int64):
                    out.global_start[i] = out.slice[i]
                    out.global_stop[i] = out.slice[i] + 1
                elif out.slice[i] is not None:
                    if out.slice[i].start is not None:
                        out.global_start[i] = self.global_start[i] + out.slice[i].start
                    if out.slice[i].stop is not None:
                        # Count forward from global_start, or backward from global_stop
                        out.global_stop[i] = self.global_start[i] + out.slice[i].stop if out.slice[i].stop > 0 else self.global_stop[i] + out.slice[i].stop
                # Revise/reset size
                out.N[i+1] = out.global_stop[i] - out.global_start[i]
            # Reset GN
            out.GN = out.N + (out.N > 1) * 2*out.NG
            # Finally, slice the caches with the revised slice
            # Except make sure they get their own memory, grids don't like to be re-used otherwise
            for cache in ('gdet', 'lapse'):
                if cache in self.__dict__:
                    # Slice over all locations in 1st index
                    out.__dict__[cache] = self.__dict__[cache][(slice(None),) + out.slice]
                    if len(out.__dict__[cache].shape) < 4:
                        out.__dict__[cache] = out.__dict__[cache][Ellipsis, np.newaxis]
            for cache in ('gcon', 'gcov', 'conn'):
                if cache in self.__dict__:
                    # Slice over all loc + 2 indices in gcon/gcov, 3 indices in conn
                    out.__dict__[cache] = self.__dict__[cache][(slice(None), slice(None), slice(None)) + out.slice]
                    if len(out.__dict__[cache].shape) < 6:
                        out.__dict__[cache] = out.__dict__[cache][Ellipsis, np.newaxis]

            return out

        elif key in self.cache:
            # Return anything we've cached
            return self.cache[key]
        elif key[:7] == 'pcoord_':
            # Return various plotting coordinates, with caching. Doesn't seem to be faster.
            mesh = False
            native = False
            half = False
            if '_mesh' in key:
                mesh = True
            if '_native' in key:
                native = True
            if '_half' in key:
                half = True
            if 'xy' in key:
                self.cache[key] = self.get_xy_locations(mesh, native)
                return self.cache[key]
            elif 'xz' in key:
                self.cache[key] = self.get_xz_locations(mesh, native, half)
                return self.cache[key]
            else:
                raise NotImplementedError("Grid cannot return plotting values for {}".format(key))

        elif key in ['n1', 'n2', 'n3']:
            return self.NTOT[int(key[-1:])]
        elif key in ['r', 'th', 'dxdX', 'dXdx', 'dXdx_cart', 'dxdX_cart', 'dXdx_bl', 'dxdX_bl', 'gcon_ks', 'gcov_ks']:
            # These keys are symmetric in phi, so we cache/return 2D versions
            self.cache[key] = getattr(self.coords, key)(self.coord_ij())
            return self.cache[key]
        
        elif key in ['phi']:
            # phi is not symmetric in phi.  Don't cache, it's big and easy
            return getattr(self.coords, key)(self.coord_all())
        elif key  == 'r1d':
            self.cache[key] = np.squeeze(self.coords.r(self.coord(np.arange(self.GN[1]), 0, 0)))
            return self.cache[key]
        elif key  == 'th1d':
            # Return coord at outer edge for minimum cylindrification
            self.cache[key] =  np.squeeze(self.coords.th(self.coord(self.GN[1]-1, np.arange(self.GN[2]), 0)))
            return self.cache[key]
        elif key  == 'phi1d':
            self.cache[key] =  np.squeeze(self.coords.phi(self.coord(0, 0, np.arange(self.GN[3]))))
            return self.cache[key]
        elif key in ['x', 'y', 'z']:
            # none of these are phi-symmetric. Ergo, 3D
            return getattr(self.coords, 'cart_' + key)(self.coord_all())
        elif key in ['X1', 'X2']:
            return self.coord_ij()[int(key[-1:])]
        elif key in ['X3']:
            return self.coord_all()[int(key[-1:])]

        # Finally, any of our attributes.  This is to allow overriding with the above
        elif key in self.__dict__:
            # Return anything we have a member for
            return self.__dict__[key]

        else:
            raise ValueError("Grid cannot find or compute {}".format(key))
        raise ValueError("Reached end of Grid retrieval, retrieving key {}".format(key))

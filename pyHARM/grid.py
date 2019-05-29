# Module defining coordinate grids

import numpy as np

try:
    import pyopencl.array as cl_array
    import loopy as lp

    from pyHARM.loopy_tools import *
    use_2018_2()
except ModuleNotFoundError:
    #print("Loading grid.py without OpenCL array support.")
    pass

from pyHARM.defs import Loci, Slices, Shapes
from pyHARM.coordinates import Minkowski, MKS, MMKS, FMKS


def make_some_grid(type, n1=128, n2=128, n3=128, a=0, hslope=0.3, r_out=50, params=None):
    """Convenience function for generating grids with default parameters used at Illinois.
    Type should be one of 'minkowski', 'mks', 'fmks'
    Size and coordinate parameters are optional with somewhat reasonable defaults.
    """
    if params is None:
        params = {}

    params['coordinates'] = type
    params['n1tot'] = n1
    params['n2tot'] = n2
    params['n3tot'] = n3

    # Things which should ideally be optional in grid creation,
    # but are not for one reason or another
    params['ng'] = 0
    params['n_prims'] = 8

    if type == 'minkowski':
        params['x1min'] = 0
        params['x2min'] = 0
        params['x3min'] = 0
        params['x1max'] = 1
        params['x2max'] = 1
        params['x3max'] = 1
    elif 'mks' in type:
        params['a'] = a
        params['hslope'] = hslope
        params['r_out'] = r_out
        if type == 'fmks':
            params['poly_xt'] = 0.82
            params['poly_alpha'] = 14.0
            params['mks_smooth'] = 0.5

    return Grid(params)



class Grid:
    """Holds all information about the (potentially a) grid of zones:
    size, shape, zones' global locations, metric tensor
    """

    # TODO privatize lots of functions as interaction will be with member variables
    def __init__(self, params):
        """
        Initialize a Grid object.  This object divides a space of native coordinates into zones, and stores the
        local metric (and some other convenient information) at each zone.

        Primarily, this object should be used to consult the grid size/shape for global calculations, and raise and
        lower the indices of fluid 4-vectors.

        :param params: Dictionary containing the following parameters, depending on coordinate system:
        n{1,2,3}tot: total number of physical grid zones in each direction
        ng: number of "ghost" zones in each direction to add for boundary conditions/domain decomposition
        coordinates: name of the coordinate system (value in parens below)
        === For Minkowski coordinates (minkowski): ===
        x{1,2,3}min: location of nearest corner of first grid zone [0,0,0]
        x{1,2,3}max: location of farthest corner of last grid zone [n1tot, n2tot, n3tot]
        === For Modified Kerr-Schild coordinates (mks): ===
        a: black hole spin, for KS metric
        hslope: narrowing parameter for X2, with the effect of increasing the resolution near the coordinate midplane
        r_out: desired location of the outer edge of the farthest radial zone on the grid
        === For Modified Modified Kerr-Schild coordinates (mmks): ===
        All of the MKS parameters, plus
        poly_xt, poly_alpha: See HARM docs wiki
        === For "Funky" Modified Kerr-Schild coordinates (fmks): ===
        All of the MMKS parameters, plus
        mks_smooth: See HARM docs wiki
        """
        # Set the basic grid parameters
        # Total grid size (all MPI processes)
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
        self.NG = params['ng']
        # Size of grid in-memory (with ghost zones)
        self.GN = self.N + 2*self.NG

        # Slices and shapes sub-objects: to hold numbers in convenient namespaces
        self.slices = Slices(self.NG)
        self.shapes = Shapes(self, params)

        if params['coordinates'] == "minkowski":
            # There are no parameters to Minkowski coordinates, so this is a class, not an object
            self.coords = Minkowski
        # MKS, FMKS expect the parameters a, hslope
        elif params['coordinates'] == "fmks":
            # FMKS additionally requires poly_xt, poly_alpha, mks_smooth
            self.coords = FMKS(params)
        elif params['coordinates'] == "mmks":
            # MMKS additionally requires poly_xt and poly_alpha
            self.coords = MMKS(params)
        elif params['coordinates'] == "mks":
            self.coords = MKS(params)
        else:
            raise ValueError("metric is {}!! must be minkowski, mks, mmks, or fmks".format(params['coordinates']))

        # Ask our new coordinate system where to start/stop the native grid,
        # so it aligns with the KS boundaries we've been assigned
        self.startx = self.coords.native_startx(params)
        if self.startx[1] < 0.0:
            raise ValueError("Not enough radial zones! Increase N1!")

        self.stopx = self.coords.native_stopx(params)

        # Finally, set up the grid
        self.dx = (self.stopx - self.startx) / self.NTOT
        self.dV = self.dx[1]*self.dx[2]*self.dx[3]

        self.gcov = np.zeros(self.shapes.locus_geom_tensor)
        self.gcon = np.zeros_like(self.gcov)

        self.gdet = np.zeros(self.shapes.locus_geom_scalar)
        self.lapse = np.zeros_like(self.gdet)

        for loc in Loci:
            ilist = np.arange(self.GN[1])
            jlist = np.arange(self.GN[2])
            x = self.coord(ilist, jlist, 0, loc)

            # Save zone centers to calculate connection coefficients
            if loc == Loci.CENT:
                x_cent = x

            gcov_loc = self.coords.gcov(x)
            gcon_loc = self.coords.gcon(gcov_loc)
            gdet_loc = self.coords.gdet(gcov_loc)
            self.gcov[loc.value] = gcov_loc
            self.gcon[loc.value] = gcon_loc
            self.gdet[loc.value] = gdet_loc
            self.lapse[loc.value] = 1./np.sqrt(-gcon_loc[0, 0])

        self.conn = self.coords.conn_func(x_cent)

        # Get loopy-based functions if we were handed an OpenCL queue
        if 'queue' in params:
            self.queue = params['queue']
            self._compile_kernels()
            self.use_ocl = True
        else:
            self.use_ocl = False

    def coord(self, i, j, k, loc=Loci.CENT):
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

        return np.squeeze(np.array(np.meshgrid(x[0], x[1], x[2], x[3])))

    def coord_bulk(self, loc=Loci.CENT):
        """Return a 3D array of all position vectors X within the physical zones.
        See coord() for use.
        """
        return self.coord(np.arange(self.N[1])+self.NG,
                          np.arange(self.N[2])+self.NG,
                          np.arange(self.N[3])+self.NG, loc=loc)

    def coord_bulk_mesh(self):
        """Returns zone corners for plotting a variable in the bulk
        """
        return self.coord(np.arange(self.N[1]+1)+self.NG,
                          np.arange(self.N[2]+1)+self.NG,
                          np.arange(self.N[3]+1)+self.NG, loc=Loci.CORN)

    def coord_all(self, loc=Loci.CENT):
        """Like coord_bulk, but including ghost zones"""
        return self.coord(np.arange(self.GN[1]),
                          np.arange(self.GN[2]),
                          np.arange(self.GN[3]), loc=loc)

    def coord_all_mesh(self):
        """Like coord_bulk_mesh, but including ghost zones"""
        return self.coord(np.arange(self.GN[1]+1),
                          np.arange(self.GN[2]+1),
                          np.arange(self.GN[3]+1), loc=Loci.CORN)

    def lower_grid(self, vcon, loc=Loci.CENT, ocl=True, out=None):
        """Lower a grid of contravariant rank-1 tensors to covariant ones."""
        if self.use_ocl and ocl:
            if out is None:
                if isinstance(vcon, np.ndarray):
                    out = np.zeros_like(vcon)
                else:
                    out = cl_array.zeros_like(vcon)
            evt, _ = self.dot2geom(self.queue, g=self.gcov_d[loc.value], v=vcon, out=out)
            return out
        else:
            # TODO support out=
            return np.einsum("ij...,j...->i...", self.gcov[loc.value, :, :, :, :, None], vcon)

    def raise_grid(self, vcov, loc=Loci.CENT, ocl=True, out=None):
        """Raise a grid of covariant rank-1 tensors to contravariant ones."""
        if self.use_ocl and ocl:
            if out is None:
                if isinstance(vcov, np.ndarray):
                    out = np.zeros_like(vcov)
                else:
                    out = cl_array.zeros_like(vcov)
            evt, _ = self.dot2geom(self.queue, g=self.gcon_d[loc.value], v=vcov, out=out)
            return out
        else:
            return np.einsum("ij...,j...->i...", self.gcon[loc.value, :, :, :, :, None], vcov)

    def dot(self, ucon, ucov, ocl=True, out=None):
        """Inner product along first index. Exists to make other code OpenCL-agnostic"""
        if self.use_ocl and ocl:
            if out is None:
                if isinstance(ucon, np.ndarray):
                    out = np.zeros(tuple(ucon.shape[1:]), ucon.dtype)
                else:
                    out = cl_array.zeros(self.queue, tuple(ucon.shape[1:]), ucon.dtype)
            evt, _ = self.dot1(self.queue, a=ucon, b=ucov, out=out)
            return out
        else:
            # TODO support out=
            return np.einsum("i...,i...", ucon, ucov)

    def _compile_kernels(self):
        # OpenCL kernels for operations that would just be broadcast in numpy,
        # and backing implementations for the wrappers above.

        # Dot two grid vectors together: sum first index
        self.dot1 = lp.make_kernel(self.shapes.isl_grid_vector,
                                     """out[i,j,k] = sum(mu, a[mu,i,j,k] * b[mu,i,j,k])""",
                                     default_offset=lp.auto)
        self.dot1 = tune_grid_kernel(self.dot1)
        # Dot to a geometry (2D) variable
        self.dot2geom = lp.make_kernel(self.shapes.isl_grid_tensor,
                                       """out[nu,i,j,k] = sum(mu, g[mu,nu,i,j] * v[mu,i,j,k])""",
                                       default_offset=lp.auto)
        # TODO is it still more efficient to break up some over k? Prefetch/explicit cache?
        self.dot2geom = tune_geom_kernel(self.dot2geom, self.shapes.grid_tensor)
        self.dot2geom2 = lp.make_kernel(self.shapes.isl_grid_tensor,
                                        """out[i,j,k] = sum(mu, sum(nu, g[mu,nu,i,j] * u[mu,i,j,k] * v[nu,i,j,k]))""",
                                        default_offset=lp.auto)
        self.dot2geom2 = tune_geom_kernel(self.dot2geom2, self.shapes.grid_tensor)

        self.dot2D2geom = lp.make_kernel(self.shapes.isl_grid_tensor,
                                        """out[i,j,k] = sum(mu, sum(nu, g[mu,nu,i,j] * u[mu, nu,i,j,k]))""",
                                        default_offset=lp.auto)
        self.dot2D2geom = tune_geom_kernel(self.dot2D2geom, self.shapes.grid_tensor)


        # Define broadcasts to and from geometry variables
        elementwise_geom_op = """out[i,j,k] = u[i,j,k] <OPERATION> g[i,j]"""
        self.timesgeom = lp.make_kernel(self.shapes.isl_grid_scalar, elementwise_geom_op.replace("<OPERATION>", "*"),
                                        default_offset=lp.auto)
        self.timesgeom = tune_geom_kernel(self.timesgeom)

        self.divbygeom = lp.make_kernel(self.shapes.isl_grid_scalar, elementwise_geom_op.replace("<OPERATION>", "/"),
                                        default_offset=lp.auto)
        self.divbygeom = tune_geom_kernel(self.divbygeom)

        vec_elementwise_geom_op = """out[mu,i,j,k] = u[mu,i,j,k] <OPERATION> g[i,j]"""
        self.vectimesgeom = lp.make_kernel(self.shapes.isl_grid_vector,
                                           vec_elementwise_geom_op.replace("<OPERATION>", "*"),
                                           default_offset=lp.auto)
        self.vectimesgeom = tune_geom_kernel(self.vectimesgeom)

        self.vecdivbygeom = lp.make_kernel(self.shapes.isl_grid_vector,
                                           vec_elementwise_geom_op.replace("<OPERATION>", "/"),
                                           default_offset=lp.auto)
        self.vecdivbygeom = tune_geom_kernel(self.vecdivbygeom)

        # Move everything we'll use to the device for convenience
        self.gcon_d = cl_array.to_device(self.queue, self.gcon)
        self.gcov_d = cl_array.to_device(self.queue, self.gcov)
        self.gdet_d = cl_array.to_device(self.queue, self.gdet)
        self.lapse_d = cl_array.to_device(self.queue, self.lapse)
        self.dx_d = cl_array.to_device(self.queue, self.dx)


    # TODO unbork
    def dt_light(self):
        """Returns the light crossing time of the smallest zone in the grid"""
        # Following stolen from bhlight's dt_light calculation

        dt_light_local = np.zeros(self.N[1], self.N[2])
        gcon = self.gcon
        CENT = Loci.CENT.value
        for mu in range(4):
            cplus = np.abs((-gcon[CENT, 0, mu, :, :] +
                            np.sqrt(gcon[CENT, 0, mu, :, :]**2 -
                                    gcon[CENT, mu, mu, :, :] * gcon[CENT, 0, 0, :, :])) /
                           gcon[CENT, 0, 0, :, :])

            cminus = np.abs((-gcon[CENT, 0, mu, :, :] -
                             np.sqrt(gcon[CENT, 0, mu, :, :]**2 -
                                     gcon[CENT, mu, mu, :, :] * gcon[CENT, 0, 0, :, :])) /
                            gcon[CENT, 0, 0, :, :])
            light_phase_speed = np.max(cplus, cminus)

            light_phase_speed = np.where(gcon[CENT, 0, mu, :, :] ** 2 -
                                         gcon[CENT, mu, mu, :, :] * gcon[CENT, 0, 0, :, :] >= 0.,
                                         light_phase_speed, 1e-20)

            dt_light_local += 1. / (self.dx[mu] / light_phase_speed)

        dt_light_local = 1. / dt_light_local

        return np.min(dt_light_local)

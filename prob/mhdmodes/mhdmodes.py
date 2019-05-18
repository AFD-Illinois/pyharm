# Tests propagation of different modes of MHD waves for one period
# Output after duration calculated here should match initial conditions to code accuracy

import numpy as np

from bounds import set_bounds
from defs import Loci


def init(params, G, P):
    """Given a grid G and list of parameters, return starting fluid primitives"""

    # "Faux-2D" planar waves direction
    # Set to 0 for "full" 3D wave
    dir_2d = params['dir_2d']

    # Number of the mode to use:
    nmode = params['nmode']

    # Mean state
    rho0 = 1
    u0 = 1  # TODO set Un for boosted entropy
    B10 = 0  # This is set later, see below
    B20 = 0
    B30 = 0

    # Wavevector
    k1 = 2*np.pi
    k2 = 2*np.pi
    k3 = 2*np.pi
    amp = 1e-4

    # Default everything to 0
    drho = 0.
    du = 0.
    du1 = 0.
    du2 = 0.
    du3 = 0.
    dB1 = 0.
    dB2 = 0.
    dB3 = 0.

    # Eigenmode definitions
    if dir_2d == 0:
        # 3D (1,1,1) wave
        B10 = 1.
        B20 = 0.
        B30 = 0.
        if nmode == 0:  # Entropy
            omega = 2.*np.pi/5.*1j
            drho = 1.
        elif nmode == 1:  # Slow
            omega = 2.35896379113*1j
            drho = 0.556500332363
            du = 0.742000443151
            du1 = -0.282334999306
            du2 = 0.0367010491491
            du3 = 0.0367010491491
            dB1 = -0.195509141461
            dB2 = 0.0977545707307
            dB3 = 0.0977545707307
        elif nmode == 2:  # Alfven
            omega = - 3.44144232573*1j
            du2 = -0.339683110243
            du3 = 0.339683110243
            dB2 = 0.620173672946
            dB3 = -0.620173672946
        else:  # Fast
            omega = 6.92915162882*1j
            drho = 0.481846076323
            du = 0.642461435098
            du1 = -0.0832240462505
            du2 = -0.224080007379
            du3 = -0.224080007379
            dB1 = 0.406380545676
            dB2 = -0.203190272838
            dB3 = -0.203190272838
    else:
        # 2D (1,1,0), (1,0,1), (0,1,1) wave
        # Constant field direction
        if dir_2d == 1:
            B20 = 1.
        elif dir_2d == 2:
            B30 = 1.
        elif dir_2d == 3:
            B10 = 1.

        if nmode == 0:  # Entropy
            omega = 2.*np.pi/5.*1j
            drho = 1.
        elif nmode == 1:  # Slow
            omega = 2.41024185339*1j
            drho = 0.558104461559
            du = 0.744139282078
            if dir_2d == 1:
                du2 = -0.277124827421
                du3 = 0.0630348927707
                dB2 = -0.164323721928
                dB3 = 0.164323721928
            elif dir_2d == 2:
                du3 = -0.277124827421
                du1 = 0.0630348927707
                dB3 = -0.164323721928
                dB1 = 0.164323721928
            elif dir_2d == 3:
                du1 = -0.277124827421
                du2 = 0.0630348927707
                dB1 = -0.164323721928
                dB2 = 0.164323721928
      
        elif nmode == 2:  # Alfven
            omega = 3.44144232573*1j
            if dir_2d == 1:
                du1 = 0.480384461415
                dB1 = 0.877058019307
            elif dir_2d == 2:
                du2 = 0.480384461415
                dB2 = 0.877058019307
            elif dir_2d == 3:
                du3 = 0.480384461415
                dB3 = 0.877058019307
      
        else:  # Fast
            omega = 5.53726217331*1j
            drho = 0.476395427447
            du = 0.635193903263
            if dir_2d == 1:
                du2 = -0.102965815319
                du3 = -0.316873207561
                dB2 = 0.359559114174
                dB3 = -0.359559114174
            elif dir_2d == 2:
                du3 = -0.102965815319
                du1 = -0.316873207561
                dB3 = 0.359559114174
                dB1 = -0.359559114174
            elif dir_2d == 3:
                du1 = -0.102965815319
                du2 = -0.316873207561
                dB1 = 0.359559114174
                dB2 = -0.359559114174

    # Override tf and the dump and log intervals, since we need exactly one cycle
    params['tf'] = tf = 2.*np.pi/np.abs(np.imag(omega))
    params['dump_cadence'] = tf/5.

    print("""
Initializing {} {} wave:
    drho = {}
    du = {}
    du1 = {}
    du2 = {}
    du3 = {}
    dB1 = {}
    dB2 = {}
    dB3 = {}""".format(["3D", "2D_1", "2D_2", "2D_3"][dir_2d], ["Entropy", "Slow", "Alfven", "Fast"][nmode],
                       drho, du, du1, du2, du3, dB1, dB2, dB3))

    x = G.coord_bulk(Loci.CENT)

    if dir_2d == 1:
        mode = amp*np.cos(k1*x[2] + k2*x[3])
    elif dir_2d == 2:
        mode = amp*np.cos(k1*x[1] + k2*x[3])
    elif dir_2d == 3:
        mode = amp*np.cos(k1*x[1] + k2*x[2])
    else:
        mode = amp*np.cos(k1*x[1] + k2*x[2] + k3*x[3])

    s = G.slices
    # Coordinates automatically suppress length-1 indices.
    # Restore them to copy to 3D array
    if len(mode.shape) < 2:
        slc = (s.a, None, None)
    elif len(mode.shape) < 3:
        slc = (s.a, s.a, None)
    else:
        slc = (s.a, s.a, s.a)

    P[s.RHO+s.bulk] = (rho0 + np.real(drho*mode))[slc]
    P[s.UU+s.bulk] = (u0 + np.real(du*mode))[slc]
    P[s.U1+s.bulk] = (np.real(du1*mode))[slc]
    P[s.U2+s.bulk] = (np.real(du2*mode))[slc]
    P[s.U3+s.bulk] = (np.real(du3*mode))[slc]
    P[s.B1+s.bulk] = (B10 + np.real(dB1*mode))[slc]
    P[s.B2+s.bulk] = (B20 + np.real(dB2*mode))[slc]
    P[s.B3+s.bulk] = (B30 + np.real(dB3*mode))[slc]

    set_bounds(params, G, P)

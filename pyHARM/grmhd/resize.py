
import numpy as np

from pyHARM.defs import Var, Loci

def resize(params, G, P, n1, n2, n3, method='linear'):
    """Resize the primitives P onto a new grid.
    Note this doesn't yet support ghost zones
    """

    P_out = np.zeros((n1, n2, n3))

    # Particle number flux
    flux[s.RHO + slc] = P[s.RHO + slc] * D['ucon'][dir][slc]

    # MHD stress-energy tensor w/ first index up, second index down
    T = Tmhd(params, G, P, D, dir, slc)
    flux[s.UU + slc] = T[0][slc] + flux[s.RHO + slc]
    flux[s.U3VEC + slc] = T[s.VEC3 + slc]

    # Dual of Maxwell tensor
    flux[s.B1 + slc] = (D['bcon'][1] * D['ucon'][dir] - D['bcon'][dir] * D['ucon'][1])[slc]
    flux[s.B2 + slc] = (D['bcon'][2] * D['ucon'][dir] - D['bcon'][dir] * D['ucon'][2])[slc]
    flux[s.B3 + slc] = (D['bcon'][3] * D['ucon'][dir] - D['bcon'][dir] * D['ucon'][3])[slc]

    if 'electrons' in params and params['electrons']:
        flux[s.KEL] = flux[s.RHO]*P[s.KEL]
        flux[s.KTOT] = flux[s.RHO]*P[s.KTOT]

    flux[s.allv + slc] *= G.gdet[loc.value][gslc]
    return flux

def interpolate(var, )
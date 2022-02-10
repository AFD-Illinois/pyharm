
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pyHARM.defs import Loci
from pyHARM.grid import Grid

def resize(params, G, P, n1, n2, n3, method='linear'):
    """Resize the primitives P onto a new grid.
    Note this doesn't yet support ghost zones
    """
    nvar = P.shape[0]
    X = G.coord_all()

    Pnew = np.zeros((nvar, n1, n2, n3))
    params_new = params.copy()
    params_new['n1tot'] = params_new['n1'] = n1
    params_new['n2tot'] = params_new['n2'] = n2
    params_new['n3tot'] = params_new['n3'] = n3
    Gnew = Grid(params_new)
    Xnew = Gnew.coord_all()

    print(X[1][:,0,0], Xnew[1][:,0,0])

    for var in range(nvar):
        interp = RegularGridInterpolator((X[1][:,0,0], X[2][0,:,0], X[3][0,0,:]), P[var], method=method, bounds_error=False)
        points = interp(Xnew[1:].T)
        Pnew[var] = points.T

    return params_new, Gnew, Pnew
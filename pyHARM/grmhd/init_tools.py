
import numpy as np

def set_fourvel_t(gcov, ucon):
    AA = gcov[0][0]
    BB = 2. * (gcov[0][1] * ucon[1] + \
               gcov[0][2] * ucon[2] + \
               gcov[0][3] * ucon[3])
    CC = 1. + gcov[1][1] * ucon[1] * ucon[1] + \
         gcov[2][2] * ucon[2] * ucon[2] + \
         gcov[3][3] * ucon[3] * ucon[3] + \
         2. * (gcov[1][2] * ucon[1] * ucon[2] + \
               gcov[1][3] * ucon[1] * ucon[3] + \
               gcov[2][3] * ucon[2] * ucon[3])

    discr = BB * BB - 4. * AA * CC
    ucon[0] = (-BB - np.sqrt(discr)) / (2. * AA)

def fourvel_to_prim(gcon, ucon, u_prim):
    alpha2 = -1.0 / gcon[0][0]
    # Note gamma/alpha is ucon[0]
    u_prim[1] = ucon[1] + ucon[0] * alpha2 * gcon[0][1]
    u_prim[2] = ucon[2] + ucon[0] * alpha2 * gcon[0][2]
    u_prim[3] = ucon[3] + ucon[0] * alpha2 * gcon[0][3]
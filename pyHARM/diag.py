# Normal diagnostics to output during every run

import numpy as np

from pyHARM.defs import Loci, Slices


def get_log_line(G, P, t):
    return "t = {}. Max divB: {}".format(t, np.max(divB(G, P)))


def divB(G, P):
    gdet = G.gdet[Loci.CENT.value]

    # If we don't have ghost zones, make our own slices
    if G.NG > 0:
        s = G.slices
    else:
        s = Slices(ng=1)

    divB = np.abs(0.25 * (
            P[s.B1][s.b, s.b, s.b] * gdet[s.b, s.b, None]
            + P[s.B1][s.b, s.l1, s.b] * gdet[s.b, s.l1, None]
            + P[s.B1][s.b, s.b, s.l1] * gdet[s.b, s.b, None]
            + P[s.B1][s.b, s.l1, s.l1] * gdet[s.b, s.l1, None]
            - P[s.B1][s.l1, s.b, s.b] * gdet[s.l1, s.b, None]
            - P[s.B1][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, None]
            - P[s.B1][s.l1, s.b, s.l1] * gdet[s.l1, s.b, None]
            - P[s.B1][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, None]
    ) / G.dx[1] + 0.25 * (
                          P[s.B2][s.b, s.b, s.b] * gdet[s.b, s.b, None]
                          + P[s.B2][s.l1, s.b, s.b] * gdet[s.l1, s.b, None]
                          + P[s.B2][s.b, s.b, s.l1] * gdet[s.b, s.b, None]
                          + P[s.B2][s.l1, s.b, s.l1] * gdet[s.l1, s.b, None]
                          - P[s.B2][s.b, s.l1, s.b] * gdet[s.b, s.l1, None]
                          - P[s.B2][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, None]
                          - P[s.B2][s.b, s.l1, s.l1] * gdet[s.b, s.l1, None]
                          - P[s.B2][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, None]
                  ) / G.dx[2] + 0.25 * (
                          P[s.B3][s.b, s.b, s.b] * gdet[s.b, s.b, None]
                          + P[s.B3][s.b, s.l1, s.b] * gdet[s.b, s.l1, None]
                          + P[s.B3][s.l1, s.b, s.b] * gdet[s.l1, s.b, None]
                          + P[s.B3][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, None]
                          - P[s.B3][s.b, s.b, s.l1] * gdet[s.b, s.b, None]
                          - P[s.B3][s.b, s.l1, s.l1] * gdet[s.b, s.l1, None]
                          - P[s.B3][s.l1, s.b, s.l1] * gdet[s.l1, s.b, None]
                          - P[s.B3][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, None]
                  ) / G.dx[3])
    if G.NG > 0:
        return divB
    else:
        divB_full = np.zeros(G.shapes.grid_scalar)
        divB_full[1:-1, 1:-1, 1:-1] += divB
        return divB_full

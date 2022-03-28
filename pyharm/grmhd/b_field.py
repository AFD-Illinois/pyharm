import numpy as np

from pyharm.defs import Loci, Slices

"""Magnetic field tools. Currently just divB"""

# TODO flux_ct in numpy, to take a step for jcon

def divB(G, B):
    gdet = G.gdet[Loci.CENT.value]

    # If we don't have ghost zones, make our own slices
    s = Slices(ng=1)

    original_shape = B.shape

    divB = np.abs(0.25 * (
            B[0][s.b, s.b, s.b] * gdet[s.b, s.b, :]
            + B[0][s.b, s.l1, s.b] * gdet[s.b, s.l1, :]
            + B[0][s.b, s.b, s.l1] * gdet[s.b, s.b, :]
            + B[0][s.b, s.l1, s.l1] * gdet[s.b, s.l1, :]
            - B[0][s.l1, s.b, s.b] * gdet[s.l1, s.b, :]
            - B[0][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, :]
            - B[0][s.l1, s.b, s.l1] * gdet[s.l1, s.b, :]
            - B[0][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, :]
            ) / G.dx[1] + 0.25 * (
            B[1][s.b, s.b, s.b] * gdet[s.b, s.b, :]
            + B[1][s.l1, s.b, s.b] * gdet[s.l1, s.b, :]
            + B[1][s.b, s.b, s.l1] * gdet[s.b, s.b, :]
            + B[1][s.l1, s.b, s.l1] * gdet[s.l1, s.b, :]
            - B[1][s.b, s.l1, s.b] * gdet[s.b, s.l1, :]
            - B[1][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, :]
            - B[1][s.b, s.l1, s.l1] * gdet[s.b, s.l1, :]
            - B[1][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, :]
            ) / G.dx[2] + 0.25 * (
            B[2][s.b, s.b, s.b] * gdet[s.b, s.b, :]
            + B[2][s.b, s.l1, s.b] * gdet[s.b, s.l1, :]
            + B[2][s.l1, s.b, s.b] * gdet[s.l1, s.b, :]
            + B[2][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, :]
            - B[2][s.b, s.b, s.l1] * gdet[s.b, s.b, :]
            - B[2][s.b, s.l1, s.l1] * gdet[s.b, s.l1, :]
            - B[2][s.l1, s.b, s.l1] * gdet[s.l1, s.b, :]
            - B[2][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, :]
            ) / G.dx[3])

    divB_full = np.zeros(original_shape[1:])
    divB_full[s.b, s.b, s.b] = divB

    return divB_full

def divB_cons(G, B):

    s = Slices(ng=1)

    divB = np.abs(0.25 * (
            B[0][s.b, s.b, s.b]
            + B[0][s.b, s.l1, s.b]
            + B[0][s.b, s.b, s.l1]
            + B[0][s.b, s.l1, s.l1]
            - B[0][s.l1, s.b, s.b]
            - B[0][s.l1, s.l1, s.b]
            - B[0][s.l1, s.b, s.l1]
            - B[0][s.l1, s.l1, s.l1]
            ) / G.dx[1] + 0.25 * (
            B[1][s.b, s.b, s.b]
            + B[1][s.l1, s.b, s.b]
            + B[1][s.b, s.b, s.l1]
            + B[1][s.l1, s.b, s.l1]
            - B[1][s.b, s.l1, s.b]
            - B[1][s.l1, s.l1, s.b]
            - B[1][s.b, s.l1, s.l1]
            - B[1][s.l1, s.l1, s.l1]
            ) / G.dx[2] + 0.25 * (
            B[2][s.b, s.b, s.b]
            + B[2][s.b, s.l1, s.b]
            + B[2][s.l1, s.b, s.b]
            + B[2][s.l1, s.l1, s.b]
            - B[2][s.b, s.b, s.l1]
            - B[2][s.b, s.l1, s.l1]
            - B[2][s.l1, s.b, s.l1]
            - B[2][s.l1, s.l1, s.l1]
            ) / G.dx[3])

    divB_full = np.zeros(B.shape[1:])
    divB_full[s.b, s.b, s.b] = divB

    return divB_full
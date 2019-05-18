# Initial conditions for Fishbone-Moncrief torus

import numpy as np

from pyHARM.defs import Loci
from pyHARM.phys import get_state
from pyHARM.bounds import set_bounds

# Different MAD initializations
#define SANE 0
#define RYAN 1
#define R3S3 2
#define GAUSSIAN 3
#define NARAYAN 4

# Alternative normalization from HARMPI.  Dosn't seem common to use
maxr_normalization = 0

def init(params, G, P):
    # Calculate derived parameters
    a = params['a']
    l = lfish_calc(a, params['r_max'])
    kappa = 1.e-3
    s = G.slices

    # Calculate fluid values inside the torus
    r = G.C.r(G.coord_bulk())
    th = G.C.r(G.coord_bulk())

    sth = np.sin(th)
    cth = np.cos(th)

    # Calculate lnh
    DD = r**2 - 2. * r + a**2
    AA = (r**2 + a**2)**2 - DD * a**2 * sth**2
    SS = r**2 + a**2 * cth**2

    thin = np.pi / 2.
    sthin = np.sin(thin)
    cthin = np.cos(thin)

    DDin = params['r_in']**2 - 2. * params['r_in'] + a**2
    AAin = (params['r_in']**2 + a**2)**2 - DDin * a**2 * sthin**2
    SSin = params['r_in']**2 + a**2 * cthin**2

    lnh = 0.5 * np.log((1. + np.sqrt(1. + 4. * (l**2 * SS**2) * DD / (AA**2 * sth**2)))
                        / (SS * DD / AA)) \
      - 0.5 * np.sqrt(1. + 4. * (l**2 * SS**2) * DD / (AA**2 * sth**2)) \
      - 2. * a * r * l / AA \
      - (0.5 * np.log((1. + np.sqrt(1. + 4. * (l**2 * SSin**2) * DDin (AAin**2 * sthin**2))) /
                      (SSin * DDin / AAin))
          - 0.5 * np.sqrt(1. + 4. * (l**2 * SSin**2) * DDin / (AAin**2 * sthin**2))
          - 2. * a * params['r_in'] * l / AAin)

    lnh[np.where(r < params['r_in'])] = 1

    # regions outside torus
    slc = np.where(np.logical_or(lnh < 0, r < params['r_in']))
    # Nominal values real value set by fixup
    P[s.RHO, slc] = 1.e-7 * RHOMIN
    P[s.UU, slc] = 1.e-7 * UUMIN
    P[s.U1, slc] = 0.
    P[s.U2, slc] = 0.
    P[s.U3, slc] = 0.
    
    # region inside magnetized torus u^i is calculated in
    # Boyer-Lindquist coordinates, as per Fishbone & Moncrief,
    # so it needs to be transformed at the end
    slc = np.where(np.logical_or(lnh < 0, r < params['r_in']))
    hm1 = np.exp(lnh) - 1.
    rho = pow(hm1 * (gam - 1.) / (kappa * gam),
               1. / (gam - 1.))
    u = kappa * pow(rho, gam) / (gam - 1.)

    # Calculate u^phi
    expm2chi = SS * SS * DD / (AA * AA * sth * sth)
    up1 = np.sqrt((-1. + np.sqrt(1. + 4. * l * l * expm2chi)) / 2.)
    up = 2. * a * r * np.sqrt(1. + up1 * up1) / np.sqrt(AA * SS * DD) + \
                np.sqrt(SS / AA) * up1 / sth


    P[RHO] = rho
    if (rho > rhomax) rhomax = rho
    u *= (1. + params['u_jitter'] * (gsl_rng_uniform(rng) - 0.5))
    P[UU] = u
    if (u > umax && r > params['r_in']) umax = u
    P[U1] = 0.
    P[U2] = 0.
    P[U3] = up

    # Convert from 4-velocity to 3-velocity
    coord_transform(G, P)
    
    P[B1] = 0.
    P[B2] = 0.
    P[B3] = 0.

    # Find the zone in which params['rBend'] of Narayan condition resides
    # This just uses the farthest process in R
    # For /very/ large N1CPU it might fail
    # TODO only do this for NARAYAN condition
    #int iend_global = 0
    #if (global_stop[0] == N1TOT && global_start[1] == 0 && global_start[2] == 0) 
    #    iend = i_of(params['rBend'])
    #iend_global = global_start[0] + iend - NG #Translate to coordinates for uu_mid below
    #iend_global = mpi_reduce_int(iend_global)
    
    # Normalize the densities so that max(rho) = 1
    umax = mpi_max(umax)
    rhomax = mpi_max(rhomax)

    P[RHO] /= rhomax
    P[UU] /= rhomax
  
    umax /= rhomax
    rhomax = 1.
    
    
    fixup(G, S)
    set_bounds(G, S)

    # Calculate UU along midplane, propagate to all processes
    # *uu_plane_send = calloc(N1TOT,sizeof())
    
    # This relies on an even N2TOT /and/ N2CPU
    # if ((global_start[1] == N2TOT/2 || N2CPU == 1) && global_start[2] == 0) 
    #     int j_mid = N2TOT/2 - global_start[1] + NG
    #     int k = NG # Axisymmetric
    #     ILOOP 
    #         int i_global = global_start[0] + i - NG
    #         uu_plane_send[i_global] = 0.25*(P[UU, k, j_mid, i] + P[UU, k, j_mid, i-1] +
    #                       P[UU, k, j_mid-1, i] + P[UU, k, j_mid-1, i-1])
    
  

    # *uu_plane = calloc(N1TOT,sizeof())
    # mpi_reduce_vector(uu_plane_send, uu_plane, N1TOT)
    # free(uu_plane_send)

    # Find corner-centered vector potential

    # Field in disk
    rho_av = 0.25*(P[RHO] + P[RHO-1] +
                    P[RHO, k, j-1, i] + P[RHO, k, j-1, i-1])
    uu_av = 0.25*(P[UU] + P[UU-1] +
                         P[UU, k, j-1, i] + P[UU, k, j-1, i-1])

    i_global = G.global_start[0] + i - G.NG
    uu_plane_av = uu_plane[i_global]
    uu_end = uu_plane[iend_global]

    if N3 > 1:
        if params['mad_type'] == SANE:
            q = rho_av/rhomax - 0.2
            
        elif params['mad_type'] == RYAN:  # BR's smoothed poloidal in-torus
            q = pow(sin(th),3)*pow(r/params['r_in'],3.)*exp(-r/400)*rho_av/rhomax - 0.2
            
        elif params['mad_type'] == R3S3:  # Just the r^3 sin^3 th term, proposed EHT standard MAD
            q = pow(r/params['r_in'],3.)*rho_av/rhomax - 0.2
            
        elif params['mad_type'] == GAUSSIAN:  # Gaussian-strength vertical threaded field
            wid = 2 #Radius of half-maximum. Units of params['r_in']
            q = gsl_ran_gaussian_pdf((r/params['r_in'])*sin(th), wid/np.sqrt(2*log(2)))
            
        elif params['mad_type'] == NARAYAN:  # Narayan '12, Penna '12 conditions
            # Former uses rstart=25, rend=810, lam_B=25
            uc = uu_av - uu_end
            ucm = uu_plane_av - uu_end
            q = pow(np.sin(th), 3) * (uc / (ucm + SMALL) - 0.2) / 0.8
            
            #Exclude q outside torus and large q resulting from division by SMALL
            if r > params['rBend'] or r < params['rBstart'] or np.abs(q) > 1.e2:
                q = 0
        
        #if (q != 0 && th > np.pi/2-0.1 && th < np.pi/2+0.1) print("q_mid is {:.10e}", q)
        else:
            print("MAD = {} not supported!".format(params['mad_type']))
            exit(-1)
      
    else:  # TODO How about 2D?
        q = rho_av/rhomax
    

    A = np.zeros()
    
    if q > 0.:
        if params['mad_type'] == NARAYAN:  # Narayan limit uses alternating loops
            lam_B = 25
            flux_correction = np.sin( 1/lam_B * (pow(r,2./3) + 15./8*pow(r,-2./5) - pow(params['rBstart'],2./3) - 15./8*pow(params['rBstart'],-2./5)))
            q_mod = q*flux_correction
            A[i, j] = q_mod
        else: 
            A[i, j] = q

    # Calculate B-field and find bsq_max
    # Flux-ct
    P[B1] = -(A[i, j] - A[i, j + 1]
                + A[i + 1, j] - A[i + 1, j + 1]) /
                (2. * dx[2] * G.gdet[Loci.CENT.value])
    P[B2] = (A[i, j] + A[i, j + 1]
     - A[i + 1, j] - A[i + 1, j + 1]) /
     (2. * dx[1] * G.gdet[Loci.CENT, j, i])

    P[B3] = 0.

    D = get_state(G, P, Loci.CENT)
    bsq = np.sum(D['bcon']*D['bcov'], axis=0)
    bsq_max = np.max(bsq)
    beta = (gam - 1.) * (P[UU]) / (0.5 * (bsq + SMALL))
    beta_min = np.min(beta)

    # bsq_max = mpi_max(bsq_max)
    # beta_min = mpi_min(beta_min)

    norm = 0
    if not maxr_normalization:
        # Ratio of max UU, beta
        beta_act = (gam - 1.) * umax / (0.5 * bsq_max)
        norm = np.sqrt(beta_act / params['beta'])
    else:
        # params['beta']_min = 100 normalization
        norm = np.sqrt(beta_min / params['beta'])
  

    # Apply normalization
    P[B1] *= norm
    P[B2] *= norm
    # B3 is uniformly 0

  
    if params['electrons']:
        #init_electrons(G,S)
        pass

    # Enforce boundary conditions
    #fixup(G, P)
    set_bounds(G, P)



def lfish_calc(a, r):

  return (((pow(a, 2) - 2. * a * np.sqrt(r) + pow(r, 2)) *
     ((-2. * a * r *
       (pow(a, 2) - 2. * a * np.sqrt(r) +
        pow(r,
      2))) / np.sqrt(2. * a * np.sqrt(r) + (-3. + r) * r) +
      ((a + (-2. + r) * np.sqrt(r)) * (pow(r, 3) + pow(a, 2) *
      (2. + r))) / np.sqrt(1 + (2. * a) / pow (r, 1.5) - 3. / r)))
    / (pow(r, 3) * np.sqrt(2. * a * np.sqrt(r) + (-3. + r) * r) *
       (pow(a, 2) + (-2. + r) * r))
      )

def add_bh_flux():
    """This seeds a central flux to the BH surface based on the parameter bh_flux
    Untested even in C HARM
    """
    # A[i, j] = 0.
    #
    # x = r*sin(th)
    # z = r*cos(th)
    # a_hyp = 20.
    # b_hyp = 60.
    # x_hyp = a_hyp*np.sqrt(1. + pow(z/b_hyp,2))
    #
    # q = (pow(x,2) - pow(x_hyp,2))/pow(x_hyp,2)
    # if x < x_hyp:
    #     A[i, j] = 10.*q

    # # Evaluate net flux
    # Phi_proc = 0.
    # for i in etc:
    #     for j in etc:
    #         jglobal = j - NG + global_start[1]
    #         #j = N2/2+NG
    #         k = NG
    #         if jglobal == N2TOT / 2:
    #             if r < params['r_in']:
    #                 B2net = (A[i, j] + A[i, j + 1] - A[i + 1, j] - A[i + 1, j + 1])
    #                 # / (2.*dx[1]*G->gdet[CENT, j, i])
    #                 Phi_proc += fabs(B2net) * np.pi / N3CPU # * 2.*dx[1]*G->gdet[CENT, j, i]

    # #If left bound in X1.  Note different convention from bhlight!
    # if G.global_start[0] == 0:
    #     for j in etc:
    #         i = 5 + NG
    #         B1net = -(A[i, j] - A[i, j+1] + A[i+1, j] - A[i+1, j+1]) # /(2.*dx[2]*G->gdet[CENT, j, i])
    #         Phi_proc += fabs(B1net)*np.pi/N3CPU  # * 2.*dx[2]*G->gdet[CENT, j, i]
    #
    # Phi = mpi_reduce(Phi_proc)
    #
    # norm = params['bh_flux']/(Phi + SMALL)

    # Flux-ct step on above
    # P[B1] += -norm
    #     * (A[i, j] - A[i, j + 1] + A[i + 1, j] - A[i + 1, j + 1])
    #     / (2. * dx[2] * G->gdet[CENT, j, i])
    # P[B2] += norm
    #     * (A[i, j] + A[i, j + 1] - A[i + 1, j] - A[i + 1, j + 1])
    #     / (2. * dx[1] * G->gdet[CENT, j, i])
    pass

def coord_transform(G, P):
    # TODO Get coords
    a = G.C.a
    r = G.C.r(G.coord_bulk())
    s = G.slices

    ucon = np.zeros([4, *P.shape[1:]])
    ucon[1] = P[s.U1]
    ucon[2] = P[s.U2]
    ucon[3] = P[s.U3]

    AA = G.gcov[0, 0]
    BB = 2. * (G.gcov[0, 1] * ucon[1] +
               G.gcov[0, 2] * ucon[2] +
               G.gcov[0, 3] * ucon[3])
    CC = 1. + \
    G.gcov[1, 1] * ucon[1]**2 + \
    G.gcov[2, 2] * ucon[2]**2 + \
    G.gcov[3, 3] * ucon[3]**2 + \
    2. * (G.gcov[1, 2] * ucon[1] * ucon[2] +
          G.gcov[1, 3] * ucon[1] * ucon[3] +
          G.gcov[2, 3] * ucon[2] * ucon[3])

    discr = BB * BB - 4. * AA * CC
    ucon[0] = (-BB - np.sqrt(discr)) / (2. * AA)
    # This is ucon in BL coords

    # transform to Kerr - Schild
    # Make transform matrix
    trans = np.zeros([4, 4, *P.shape[1:]])
    for mu in range(4):
        trans[mu, mu] = 1.
    
    trans[0, 1] = 2 * r / (r**2 - 2*r + a**2)
    trans[3, 1] = a / (r**2 - 2*r + a**2)

    # This is ucon in KS coords
    ucon = np.einsum("ij...,j...->i...",trans,ucon)

    # Transform to MKS or MMKS
    # TODO probs needs to be flipped. Or add dxdX_inverse to coordinate functions
    invtrans = G.C.dxdX()
    np.linalg.inv(invtrans, trans)

    ucon = np.einsum("ij...,j...->i...",trans,ucon)

    # Solve for v.Use same u ^ t, unchanged under KS -> KS'
    alpha = G.lapse[Loci.CENT.value]
    gamma = ucon[0] * alpha
    
    beta[1] = alpha**2 * G.gcon[Loci.CENT, 0, 1]
    beta[2] = alpha**2 * G.gcon[Loci.CENT, 0, 2]
    beta[3] = alpha**2 * G.gcon[Loci.CENT, 0, 3]
    
    P[s.U1] = ucon[1] + beta[1] * gamma / alpha
    P[s.U2] = ucon[2] + beta[2] * gamma / alpha
    P[s.U3] = ucon[3] + beta[3] * gamma / alpha
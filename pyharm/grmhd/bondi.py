__license__ = """
 File: bondi.py
 
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
from scipy import optimize
from scipy.interpolate import splrep, splev
from scipy.integrate import odeint, solve_ivp

from pyharm.defs import Loci

__doc__ = \
"""Compute the Bondi solution, and viscous extensions.
Originally from a script by Vedant Dhruv
"""

############### COMPUTE ANALYTIC IDEAL BONDI SOLUTION ###############

# Nonlinear expression to solve for T
def T_func(T, r, C3, C4, N):
    return (1 + (1 + N/2)*T)**2 * (1 - 2./r + (C4**2/(r**4 * T**N))) - C3

# Obtain primitives for Bondi problem
def get_bondi_soln(mdot, rc, gam, grid, add_fourv=True, add_dP=True):
    N    = 2./ (gam - 1)
    vc   = np.sqrt(1. / (2 * rc))
    csc  = np.sqrt(vc**2 / (1 - 3*vc**2))
    Tc   = 2*N*csc**2 / ((N + 2)*(2 - N*csc**2))
    C4   = Tc**(N/2)*vc*rc**2
    C3   = (1 + (1 + N/2)*Tc)**2 * (1 - 2./rc + vc**2)

    # Root find T
    T = np.zeros_like(grid['r1d'])
    for index, r in enumerate(grid['r1d']):
        T0       = Tc
        sol      = optimize.root(T_func, [T0], args=(r, C3, C4, N))
        T[index] = sol.x[0]
        if (sol.success!=True):
            print("Not converged at r = {:.2f}", r)

    # Compute remaining fluid variables
    soln = {}
    soln['T'] = T
    soln['v'] = -C4 / (T**(N/2) * grid['r1d']**2)
    soln['K'] = (4*np.pi*C4 / mdot) ** (2./N)

    soln['rho'] = soln['K']**(-N/2) * T**(N/2)
    soln['u']   = (N/2) * soln['K']**(-N/2) * T**(N/2 + 1)

    soln['mdot'] = mdot
    soln['N']    = N
    soln['rc']   = rc

    if add_fourv:
        compute_ub(soln, grid)
    
    if add_dP:
        compute_dP0(soln, grid)

    return soln

# Compute four vectors
def compute_ub(soln, grid):

    # We have u^r in BL. We need to convert this to ucon in MKS
    # First compute u^t in BL
    ucon_bl = np.zeros((4, grid['n1'], grid['n2'], 1), dtype=float)
    gcov_bl = grid['gcov_bl']
    AA = gcov_bl[0,0]
    BB = 2. * gcov_bl[0,1]*soln['v'][:,None]
    CC = 1. + gcov_bl[1,1]*soln['v'][:,None]**2
    
    discr = BB*BB - 4.*AA*CC
    ucon_bl[0] = (-BB - np.sqrt(discr)) / (2.*AA)
    ucon_bl[1] = soln['v'][:,None]

    # Convert ucon(Bl) to ucon(KS)
    dxdX = np.zeros((4, 4, grid['n1'], grid['n2'], 1), dtype=float)
    dxdX[0,0] = dxdX[1,1] = dxdX[2,2] = dxdX[3,3] = 1.
    dxdX[0,1] = 2*grid['r'] / (grid['r']**2 - 2.*grid['r'] + grid['a']**2)
    dxdX[3,1] = grid['a']/(grid['r']**2 - 2.*grid['r'] + grid['a']**2)

    ucon_ks = np.zeros_like(ucon_bl)
    for mu in range(4):
        for nu in range(4):
            ucon_ks[mu] += dxdX[mu,nu] * ucon_bl[nu]

    # Convert ucon(KS) to ucon(MKS/FMKS)
    ucon_mks = np.zeros_like(ucon_bl)
    dxdX = grid.coords.dxdX(grid.coord_ij())
    for mu in range(4):
        for nu in range(4):
            ucon_mks[mu] += dxdX[mu,nu] * ucon_ks[nu]

    gcov = grid['gcov'][Loci.CENT.value]
    ucov_mks = np.einsum('mn...,n...->m...', gcov, ucon_mks)

    # Compute velocity primitives
    utilde = np.zeros((3, grid['n1'], grid['n2'], 1), dtype=float)

    gcon = grid['gcon'][Loci.CENT.value]
    alpha = 1./np.sqrt(-gcon[0,0])
    beta  = np.zeros_like(utilde)
    beta[0] = alpha * alpha * gcon[0,1]
    beta[1] = alpha * alpha * gcon[0,2]
    beta[2] = alpha * alpha * gcon[0,3]
    gamma = ucon_mks[0] * alpha

    utilde[0] = ucon_mks[1] + beta[0]*gamma/alpha
    utilde[1] = ucon_mks[2] + beta[1]*gamma/alpha
    utilde[2] = ucon_mks[3] + beta[2]*gamma/alpha

    # compute magnetic 4-vector
    B = np.zeros_like(utilde)
    # radial magnetic field (B1 = 1/r^3)
    B[0] = 1. / grid['r']**3

    lapse = grid['lapse'][Loci.CENT.value]
    gti    = gcon[0,1:4]
    gij    = gcov[1:4,1:4]
    beta_i = np.einsum('si...,i...->si...', gti, lapse**2)
    qsq    = np.einsum('yi...,yi...->i...', np.einsum('xy...,x...->y...', gij, utilde), utilde)
    gamma  = np.sqrt(1 + qsq)
    ui     = utilde - np.einsum('si...,i...->si...', beta_i, gamma/lapse)
    ut     = gamma/lapse

    bt = np.einsum('yi...,yi...->i...', np.einsum('sm...,s...->m...', gcov[1:4,:], B), ucon_mks)
    bi = (B + np.einsum('si...,i...->si...', ucon_mks[1:4], bt)) / ucon_mks[0,None]
    bcon_mks = np.append(bt[None], bi, axis=0)
    bcov_mks = np.einsum('mn...,n...->m...', gcov, bcon_mks)

    soln['ucon'] = ucon_mks[:,:,0,0]
    soln['ucov'] = ucov_mks[:,:,0,0]
    soln['bcon'] = bcon_mks[:,:,0,0]
    soln['bcov'] = bcov_mks[:,:,0,0]
    soln['bsq']  = np.einsum('mi...,mi...->i...', soln['bcon'], soln['bcov'])



############### ADDITIONAL FUNCTIONS FOR VISCOUS BONDI FLOW ###############
# Compute Braginskii pressure anisotropy value
def compute_dP0(soln, grid):
    soln['tau'] = 30.
    soln['eta'] = 0.01
    nu_emhd     = soln['eta'] / soln['rho']
    dP0         = np.zeros(grid['n1'], dtype=float)

    # Compute derivatives of 4-velocity
    ducovDx1 = np.zeros((grid['n1'], 4), dtype=float) # Represents d_x1(u_\mu)
    delta = 1.e-5
    x1    = grid['X1'][:,0]
    x1h   = x1 + delta
    x1l   = x1 - delta

    ucovt_splrep = splrep(x1, soln['ucov'][0])
    ucovr_splrep = splrep(x1, soln['ucov'][1])
    ucovt_h = splev(x1h, ucovt_splrep) 
    ucovt_l = splev(x1l, ucovt_splrep) 
    ucovr_h = splev(x1h, ucovr_splrep) 
    ucovr_l = splev(x1l, ucovr_splrep)

    ducovDx1[:,0] = np.squeeze((ucovt_h - ucovt_l) / (x1h - x1l))
    ducovDx1[:,1] = np.squeeze((ucovr_h - ucovr_l) / (x1h - x1l))

    print(ducovDx1.shape)

    gcon = grid['gcon'][Loci.CENT.value]

    for mu in range(4):
        for nu in range(4):
            if mu == 1:
                dP0 += 3*soln['rho']*nu_emhd * (soln['bcon'][mu]*soln['bcon'][nu] / soln['bsq']) \
                        * ducovDx1[:,nu]
                
            gamma_term_1 = np.zeros((grid['n1'], grid['n2']), dtype=float)
            for sigma in range(4):
                gamma_term_1 += (3*soln['rho']*nu_emhd * (soln['bcon'][mu]*soln['bcon'][nu] / soln['bsq'])) \
                                * np.squeeze(-grid['conn'][sigma, mu, nu][:,0,0] * soln['ucov'][sigma])

            dP0 += np.mean(gamma_term_1, axis=1)

        derv_term_2 = np.zeros((grid['n1'], grid['n2']), dtype=float)
        if mu == 1:
            for sigma in range(4):
                derv_term_2 += (-soln['rho']*nu_emhd * ducovDx1[:,sigma]) * gcon[mu,sigma][:,0,0]

        dP0 += np.mean(derv_term_2, axis=1)

        gamma_term_2 = np.zeros((grid['n1'], grid['n2']), dtype=float)
        for sigma in range(4):
            for delta in range(4):
                    gamma_term_2 += (soln['rho']*nu_emhd)* np.squeeze(grid['conn'][sigma, mu, delta][:,0,0] * gcon[mu, delta][:,0,0]) * soln['ucov'][sigma]

        dP0 += np.mean(gamma_term_2, axis=1)
    
    soln['dP0'] = dP0
    return dP0

def ddP_dX1(dP, x1, tau, ur_splrep, dP0_splrep, coeff_splrep):
    """Return derivative d(dP)/dx1. Refer Equation (36) in grim paper"""
    ur    = splev(x1, ur_splrep)
    dP0   = splev(x1, dP0_splrep)
    coeff = splev(x1, coeff_splrep)

    derivative = -((dP - dP0) / (tau * ur)) - (dP * coeff)
    return derivative

# Compute the coefficient of the second term on the RHS of the evolution equation of dP
def compute_rhs_second_term(soln, grid, gam):
    nu_emhd = soln['eta'] / soln['rho']
    P = soln['u'] * (gam - 1.)

    # compute derivative
    delta = 1.e-5
    x1    = grid['X1'][:,0,0]
    x1h   = x1 + delta
    x1l   = x1 - delta
    expr  = np.log(soln['tau'] / (soln['rho'] * nu_emhd * P))
    expr_splrep = splrep(x1, expr)
    expr_h = splev(x1h, expr_splrep)
    expr_l = splev(x1l, expr_splrep)

    coeff  = 0.5 * (expr_h - expr_l) / (x1h - x1l)

    return coeff

__license__ = """
 File: bondi.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020-2023, Ben Prather and AFD Group at UIUC
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

import copy

import numpy as np
from scipy import optimize
from scipy.interpolate import splrep, splev
from scipy.integrate import odeint, solve_ivp

from pyharm.defs import Loci
from pyharm.grid import Grid
from pyharm.grmhd.init_tools import *
from pyharm.fluid_state import FluidState
from pyharm.variables import braginskii_dP

__doc__ = \
"""Compute the Bondi solution, and viscous extensions.
Originally from a script by Vedant Dhruv
"""

############### COMPUTE ANALYTIC IDEAL BONDI SOLUTION ###############

# Nonlinear expression to solve for T
def _T_func(T, r, C3, C4, N):
    return (1 + (1 + N/2)*T)**2 * (1 - 2./r + (C4**2/(r**4 * T**N))) - C3

# Obtain primitives for Bondi problem
def get_bondi_soln(mdot, rc, gam, r_values):
    N    = 2./ (gam - 1)
    vc   = np.sqrt(1. / (2 * rc))
    csc  = np.sqrt(vc**2 / (1 - 3*vc**2))
    Tc   = 2*N*csc**2 / ((N + 2)*(2 - N*csc**2))
    C4   = Tc**(N/2)*vc*rc**2
    C3   = (1 + (1 + N/2)*Tc)**2 * (1 - 2./rc + vc**2)
    K = (4*np.pi*C4 / mdot) ** (2./N)

    # Root find T
    T = np.zeros_like(r_values)
    for index, r in enumerate(r_values):
        T0       = Tc
        sol      = optimize.root(_T_func, [T0], args=(r, C3, C4, N))
        T[index] = sol.x[0]
        if (sol.success!=True):
            print("Not converged at r = {:.2f}", r)

    # Compute remaining fluid variables
    soln = {}
    soln['T'] = T

    soln['rho'] = K**(-N/2) * T**(N/2)
    soln['u']   = (N/2) * K**(-N/2) * T**(N/2 + 1)
    soln['ur'] = -C4 / (T**(N/2) * r_values**2)

    return soln

def get_bondi_fluid_state(mdot, rc, gam, grid):
    soln = get_bondi_soln(mdot, rc, gam, grid['r1d'])

    # We have u^r in BL from the soln,
    # which we must convert to primitive U1,2,3 in native coords
    ucon_bl = np.zeros((4, grid['n1'], grid['n2'], grid['n3']), dtype=float)
    ucon_bl[1] = soln['ur'][:,None,None]
    set_fourvel_t(grid['gcov_bl'], ucon_bl)
    # Convert ucon(Bl) to ucon(KS)
    ucon_ks = np.einsum("ij...,j...->i...", grid['dxdX_bl'], ucon_bl)
    # Convert ucon(KS) to ucon(MKS/FMKS)
    ucon_mks = np.einsum("i...,ij...->j...", ucon_ks, grid['dXdx'])
    # Convert to primitive vars (TODO do I even need this?)
    utilde = fourvel_to_prim(grid['gcon'], ucon_mks)

    # Construct a fluid state object
    state_data = {}
    sizer = np.ones((grid['n1'], grid['n2'], grid['n3']))
    state_data['RHO'] = state_data['rho'] = soln['rho'][:,None,None]*sizer
    state_data['UU'] = state_data['u'] = soln['u'][:,None,None]*sizer
    state_data['U1'] = utilde[0]
    state_data['U2'] = utilde[1]
    state_data['U3'] = utilde[2]
    state_data['uvec'] = utilde
    state_data['B1'] = 1/grid['r']**3 * sizer
    state_data['B2'] = np.zeros_like(state_data['B1'])
    state_data['B3'] = np.zeros_like(state_data['B1'])
    state_data['B'] = np.array([state_data['B1'], state_data['B2'], state_data['B3']])
    # For good measure
    state_data['ur'] = soln['ur']

    # Add the parameters.
    # We need a superset of grid parameters to make most things work
    params = {**grid.params}
    params['mdot'] = mdot
    params['rc'] = params['rs'] = rc
    params['gam'] = gam

    return FluidState(state_data, params=params, grid=grid)

############### ADDITIONAL FUNCTIONS FOR VISCOUS BONDI FLOW ###############

def _ddP_dX1(x1, dP, tau, ur_splrep, dP0_splrep, coeff_splrep):
    """Return derivative d(dP)/dx1. Refer Equation (36) in grim paper"""
    ur    = splev(x1, ur_splrep)
    dP0   = splev(x1, dP0_splrep)
    coeff = splev(x1, coeff_splrep)

    derivative = -((dP - dP0) / (tau * ur)) - (dP * coeff)
    #print("ddp eval: x1 {} dP {} ur {} dP0 {} coeff {} derivative {}".format(x1, dP, ur, dP0, coeff, derivative))
    return derivative

def compute_rhs_second_term(state):
    """Compute the coefficient of the second term on the RHS of the evolution
    equation of dP.
    """

    # compute derivative
    delta = 1.e-5
    x1    = state['X1'][:,0,0]
    x1h   = x1 + delta
    x1l   = x1 - delta
    expr  = np.log(state['tau'] / (state['eta'] * state['u'] * (state['gam'] - 1.)))
    expr_splrep = splrep(x1, expr[:,0,0])
    expr_h = splev(x1h, expr_splrep)
    expr_l = splev(x1l, expr_splrep)

    coeff  = 0.5 * (expr_h - expr_l) / (2*delta)

    return coeff

def compute_dP(mdot, rc, gam, input_grid, eta=0.01, tau=30, start=0., npoints=1000):
    # Make sure we have enough points for an accurate solution
    # grid_params = {}
    # grid_params['n1'] = npoints
    # grid_params['n2'] = 1
    # grid_params['n3'] = 1
    # for key in ['startx1', 'startx2', 'startx3', 'r_in',
    #             'r_out', 'coordinates', 'a', 'hslope']:
    #     grid_params[key] = input_grid.params[key]
    # grid = Grid(grid_params, cache_conn=True)
    grid = input_grid

    # Get the solution on the greater number of points
    state = get_bondi_fluid_state(mdot, rc, gam, grid)
    state.params['eta'] = eta
    state.params['tau'] = tau
    coeff = compute_rhs_second_term(state)

    x1 = grid['X1'][:,0,0]
    dx1 = grid['dx1']
    ur_splrep    = splrep(x1, state['ucon'][1][:,0,0])
    dP0_splrep   = splrep(x1, np.mean(state['dP0'], axis=(1,2))) # TODO call to pass eta?
    coeff_splrep = splrep(x1, coeff)

    ode_soln = solve_ivp(_ddP_dX1, (x1[-1], x1[0]), [start],
                        args=(tau, ur_splrep, dP0_splrep, coeff_splrep),
                        dense_output=True)
    return ode_soln.sol(input_grid['X1'][:,0,0])[0,:]

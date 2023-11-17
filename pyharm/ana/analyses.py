__license__ = """
 File: analyses.py
 
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

import sys

from .reductions import *
from .. import io

__doc__ = \
"""Groups of particular related reductions performed over a GRMHD run.

Each function takes a dump object and a dict 'out' to fill
when computing new variables.  This allows collation in analysis.py for the pyharm-analysis script

Variables are organized by remaining independent variable, as described in ana_results.py
"""

defaults = {'iEH': 5, 'rF': 40, 'do_tavgs': False, 'rTh': 40}

def _get(kwargs, par_name):
    if par_name in kwargs:
        return kwargs[par_name]
    else:
        return defaults[par_name]

def basic(dump, out, **kwargs):
    """Anything necessary for other functions: t, EH fluxes, whether to average
    """

    # We have r_eh from the dump, but sometimes we want to calculate
    # at a particular location
    iEH = _get(kwargs, 'iEH')

    # Record dump time. TODO fallbacks?
    out['coord/t'] = dump['t']
    # Record whether this dump is part of the average
    out['t/is_avg'] = kwargs['t_avg_start'] < dump['t'] < kwargs['t_avg_end']

    print("Analyzing t={}".format(int(dump['t'])), file=sys.stderr)

    # FIELD STRENGTH
    # The HARM B_unit is sqrt(4pi)*c*sqrt(rho), and this is standard for EHT comparisons
    out['t/Phi_b'] = 0.5 * shell_sum(dump, 'abs_B1', at_i=iEH)

    # FLUXES
    # Radial profiles of Mdot and Edot, and their particular values
    # EHT code-comparison normalization has all these values positive
    for var, flux in [['Edot', 'FE'], ['Mdot', 'FM'], ['Ldot', 'FL']]:
        out['t/'+var] = shell_sum(dump, flux, at_i=iEH)
    # Mdot and Edot are defined inward/positive at EH
    out['t/Mdot'] *= -1
    out['t/Edot'] *= -1

def dynamo(dump, out, **kwargs):
    """Compare magnetization in the upper and lower hemisphere of EH, and at 5 r_g
    """
    # We have r_eh from the dump, but sometimes we want to calculate
    # at a particular location
    iEH = _get(kwargs, 'iEH')

    # FIELD STRENGTHS
    # The HARM B_unit is sqrt(4pi)*c*sqrt(rho), and this is standard for EHT comparisons
    out['t/Phi_b_upper'] = shell_sum(dump, 'B1', at_i=iEH, j_slice=(0, dump['n2']//2))
    out['t/Phi_b_lower'] = shell_sum(dump, 'B1', at_i=iEH, j_slice=(dump['n2']//2, dump['n2']))

    out['t/Phi_b_5'] = 0.5 * shell_sum(dump, 'abs_B1', at_r=5)
    out['t/Phi_b_upper_5'] = shell_sum(dump, 'B1', at_r=5, j_slice=(0, dump['n2']//2))
    out['t/Phi_b_lower_5'] = shell_sum(dump, 'B1', at_r=5, j_slice=(dump['n2']//2, dump['n2']))

def r_profiles(dump, out, vars=('rho', 'Pg', 'u^r', 'u^3', 'u_3', 'b', 'inv_beta', 'Ptot'), **kwargs):
    """Calculate Radial profiles, by averaging over phi and some portion of theta.
    Separate averages over the comparison "disk" portion and the rest of the domain, marked "notdisk".
    Keeps time-dependent versions for averaging post-hoc
    """
    # sigma, bsq are derived afterward
    jmin, jmax = get_j_bounds(dump)
    for var in vars:
        out['rt/' + var + '_disk'] = shell_avg(dump, var, j_slice=(jmin, jmax))
        out['rt/' + var + '_notdisk'] = (shell_avg(dump, var, j_slice=(0, jmin)) + 
                                         shell_avg(dump, var, j_slice=(jmax, dump['n2']))) / 2
        if _get(kwargs, 'do_tavgs'):
            out['r/' + var + '_disk'] = out['rt/' + var + '_disk']
            out['r/' + var + '_notdisk'] = out['rt/' + var + '_notdisk']

def r_profiles_cc(dump, out, **kwargs):
    """Radial profiles of everything used in the MAD Code Comparison '22
    """
    r_profiles(dump, out, ('rho', 'Pg', 'u^r', 'u^th', 'u^3', 'b^r', 'b^th', 'b^3', 'b', 'inv_beta', 'Ptot'), **kwargs)

def r_flux_profiles(dump, out, vars=('FM', 'FE', 'FL'), **kwargs):
    """Radial profiles of conserved mass, energy, angular momentum
    """
    jmin, jmax = get_j_bounds(dump)
    for var in vars:
        out['rt/' + var + '_all'] = shell_sum(dump, var)
        out['rt/' + var + '_disk'] = shell_sum(dump, var, j_slice=(jmin, jmax))
        if _get(kwargs, 'do_tavgs'):
            out['r/' + var + '_all'] = out['rt/' + var + '_all']
            out['r/' + var + '_disk'] = out['rt/' + var + '_disk']

def slices(dump, out, **kwargs):
    """Capture midplane, poloidal, and accretion-region slices of the original primitive variables.
    """

    out['thphit/i5.prims'] = np.squeeze(dump[5, :, :]['prims'])
    i5m = i_of(dump['r1d'], 5)
    out['thphit/5M.prims'] = np.squeeze(dump[i5m, :, :]['prims'])

    out['rtht/i0.prims'] = np.squeeze(dump[:, :, 0]['prims'])
    out['rtht/iN_2.prims'] = np.squeeze(dump[:, :, dump['n3']//2-1]['prims'])

    out['rphit/iN_2.prims'] = np.squeeze(dump[:, dump['n2']//2-1, :]['prims'])


def th_profiles(dump, out, vars=('inv_beta', 'sigma'), **kwargs):
    """Calculate any full-theta profiles: 5-zone radial averages starting from 'rTh' (also averaged in phi).
    Keeps time-dependent versions for averaging post-hoc
    """
    rTh = _get(kwargs, 'rTh')
    at_i = i_of(dump['r1d'], rTh)
    for var in vars:
        out['tht/' + var + '_' + str(int(rTh))] = theta_profile(dump, var, at_i, 5, fold=False)
        if _get(kwargs, 'do_tavgs'):
            out['th/' + var + '_' + str(int(rTh))] = out['tht/' + var + '_' + str(int(rTh))]

def omega_bz(dump, out, **kwargs):
    """Field rotation rate, for comparison against Blandford-Znajek prediction
    """
    at_i = _get(kwargs, 'iEH') # At event horizon
    # Average the next 5 zones out from zone 5
    out['htht/omega'] = theta_profile(dump, 'F_0_1', at_i, 5) / theta_profile(dump, 'F_1_3', at_i, 5)

    if _get(kwargs, 'do_tavgs'):
        out['hth/omega'] = out['htht/omega']

def rth_profiles(dump, out, vars=('betainv', 'rho', 'sigma', 'Theta'), **kwargs):
    """2D profiles in r/th, time-averaged.  Default betainv, rho, sigma, Theta
    """
    for var in vars:
        out['rtht/' + var] = np.mean(dump[var], axis=-1)
        if _get(kwargs, 'do_tavgs'):
            out['rth/' + var] = out['rtht/' + var]

def diagnostics(dump, out, **kwargs):
    """Energy ratios on the grid to gauge floor effectiveness, along with total floor hit and inversion flags
    """
    # Maxima (for gauging floors)
    for var in ['sigma', 'inv_beta', 'Theta', 'U']:
        out['t/' + var + '_max'] = np.max(dump[var])
    # Minima
    for var in ['rho', 'U']:
        out['t/' + var + '_min'] = np.min(dump[var])
    out['rt/total_floors'] = np.sum(dump['floors'] > 0, axis=(1,2))
    out['rt/total_fails'] = np.sum(dump['fails'] > 0, axis=(1,2))

    # This isn't useful in single-precision
    #out['t/MaxDivB'] = np.max(dump['divB'])

def print_flags(dump, out, **kwargs):
    """Just check floor and failure flags. Used as a cursory check for run sanity
    """
    total_floors = np.sum(dump['floors'] > 0, axis=(1,2))
    total_fails = np.sum(dump['fails'] > 0, axis=(1,2))
    print("{} floors: {} failures: {}".format(dump.fname, total_floors, total_fails), file=sys.stderr)
    # TODO some automated "bad" criterion

def print_divb(dump, out, **kwargs):
    """Just check divB. Used as a cursory check for restart file sanity
    """
    print("{} max div B: {}".format(dump.fname, np.max(dump['divB'])), file=sys.stderr)

def r_profile_phi(dump, out, **kwargs):
    """Spherical and midplane magnetizations of radial shells, analogous to FM or FE for Phi_b
    """
    out['rt/Phi_b_sph'] = 0.5 * shell_sum(dump, 'abs_B1')
    out['rt/Phi_b_mid'] = np.zeros_like(out['rt/Phi_b_sph'])
    for i in range(out['rt/Phi_b_mid'].shape[0]):
        out['rt/Phi_b_mid'][i] = midplane_sum(dump, -dump['B2'], r_slice=(0,i))

def madcc(dump, out, **kwargs):
    """Functions for MAD Code Comparison '22, nontrivial mandatory diagnostics. See that paper/doc for full descriptions
    """
    out['rt/thrho'] = (shell_sum(dump, dump['rho']*np.abs(np.pi/2 - dump['th'])) /
                        shell_sum(dump, dump['rho']))

    if _get(kwargs, 'do_tavgs'):
        for var in ('rho', 'u^r', 'u^th', 'u^3', 'b^r', 'b^th', 'b^3', 'b', 'Pg', 'inv_beta', 'sigma'):
            out['rth/' + var] = dump[var].mean(axis=-1)

def madcc_optional(dump, out, **kwargs):
    """Functions for MAD Code Comparison '22, optional diagnostics
    """
    jmin, jmax = get_j_bounds(dump)

    # Wavelength of fastest MRI mode for calculating suppression factor
    out['rt/lam_MRI'] = (shell_sum(dump, dump['rho']*dump['lam_MRI']) /
                          shell_sum(dump, dump['rho']))

    # Correlation functions at specific radii
    for var in ['rho', 'inv_beta']:
        out['phit/' + var + '_cf10'] = corr_midplane(dump[var], at_i1=i_of(dump['r1d'], 10))
        out['phit/' + var + '_cf20'] = corr_midplane(dump[var], at_i1=i_of(dump['r1d'], 20))
        out['phit/' + var + '_cf30'] = corr_midplane(dump[var], at_i1=i_of(dump['r1d'], 30))
        out['phit/' + var + '_cf50'] = corr_midplane(dump[var], at_i1=i_of(dump['r1d'], 50))

    # Jet profile moments/ellipse
    for w_r in [50, 100]:
        for w_pole, w_slice in [('north', (0, jmin)), ('south', (jmax, dump.header['n2']))]:
            # CM
            out['thphit/jet_psi_'+w_pole+'_'+str(w_r)] = dump['jet_psi'][i_of(dump['r1d'], w_r), :, :]
            out['t/M_'+w_pole+'_'+str(w_r)] = M = shell_sum(dump, dump['jet_psi'] * np.cos(dump['th']), j_slice=w_slice, at_r=w_r)
            out['t/X_'+w_pole+'_'+str(w_r)] = X = 1/M * shell_sum(dump, dump['x']*dump['jet_psi'] * np.cos(dump['th']), j_slice=w_slice, at_r=w_r)
            out['t/Y_'+w_pole+'_'+str(w_r)] = Y = 1/M * shell_sum(dump, dump['y']*dump['jet_psi'] * np.cos(dump['th']), j_slice=w_slice, at_r=w_r)
            # Moments
            out['t/Ixx_'+w_pole+'_'+str(w_r)] = shell_sum(dump, (dump['x'] - X)**2 * dump['jet_psi'] * np.cos(dump['th']), j_slice=w_slice, at_r=w_r)
            out['t/Iyy_'+w_pole+'_'+str(w_r)] = shell_sum(dump, (dump['y'] - Y)**2 * dump['jet_psi'] * np.cos(dump['th']), j_slice=w_slice, at_r=w_r)
            out['t/Ixy_'+w_pole+'_'+str(w_r)] = shell_sum(dump, (dump['x'] - X) * (dump['y'] - Y) * dump['jet_psi'] * np.cos(dump['th']), j_slice=w_slice, at_r=w_r)
            del M, X, Y

    if _get(kwargs, 'do_tavgs'):
        # Full midplane correlation function, time-averaged
        for var in ['rho', 'inv_beta']:
            out['rphi/' + var + '_cf'] = corr_midplane(dump[var])


# Polar profiles of different fluxes and variables
def jet_profile(dump, out, **kwargs):
    """Theta profiles of many (many) different variables at some particular radius. Used to gauge what is/isn't jet and where it wanders
    """
    rBZ = _get(kwargs, 'rBZ')
    iBZ = i_of(dump['r1d'], rBZ)
    s_dump = dump[iBZ]
    for var in ['rho', 'bsq', 'b^r', 'b^th', 'b^3', 'u^r', 'u^th', 'u^3', 'FM', 'FE', 'FE_EM', 'FE_Fl', 'FL', 'FL_EM', 'FL_Fl', 'betagamma', 'Be_nob', 'Be_b']:
        out['tht/' + var + '_' + str(int(rBZ))] = np.sum(s_dump[var], axis=-1)
        if _get(kwargs, 'do_tavgs'):
            out['th/' + var + '_' + str(int(rBZ))] = out['tht/' + var + '_']
            out['thphi/' + var + '_' + str(int(rBZ))] = s_dump[var]

def jet_cuts(dump, out, **kwargs):
    """Blandford-Znajek Luminosity L_BZ, calculated with a sampling of different cuts floated for EHTC V '19
    """
    # TODO cut on phi/t averages? -- needs 2-pass cut...
    cuts = {'sigma1': lambda dump: (dump['sigma'] > 1),
            'Be_b0': lambda dump: (dump['Be_b'] > 0.02),
            'Be_b1': lambda dump: (dump['Be_b'] > 1),
            'Be_nob0': lambda dump: (dump['Be_nob'] > 0.02),
            'Be_nob1': lambda dump: (dump['Be_nob'] > 1),
            #'mu1' : lambda dump : (dump['mu'] > 1),
            'bg1': lambda dump: (dump['betagamma'] > 1.0),
            'bg05': lambda dump: (dump['betagamma'] > 0.5),
            'allp': lambda dump: (dump['FE'] > 0),
            'morep': lambda dump: (np.logical_and(dump['FE_norho'] > 0,
                                    np.logical_or(dump['th'] < 1, dump['th'] > np.pi - 1)))}

    # Terminology:
    # LBZ = E&M energy only, any cut
    # Lj = full E flux, any cut
    # Ltot = Lj_allp = full luminosity wherever it is positive
    for lum, flux in [['LBZ', 'FE_EM'], ['Lj', 'FE_norho']]:
        for cut in cuts.keys():
            out['rt/' + lum + '_' + cut] = shell_sum(dump, flux, mask=cuts[cut](dump))
            out['t/' + lum + '_' + cut] = out['rt/' + lum + '_' + cut][iBZ]
            if _get(kwargs, 'do_tavgs'):
                out['r/' + lum + '_' + cut] = out['rt/' + lum + '_' + cut]

def jet_cut_lite(dump, out, **kwargs):
    """Compute jet powers with just the default cut from EHTC Paper V '19.
    These are the powers used in the table in that paper and the MAD Code Comparison '22
    """
    is_jet = dump['Be_b'] > 1
    for lum, flux in [['Mdot_jet', 'FM'], ['P_jet', 'FE'], ['P_EM_jet', 'FE_EM'], ['P_PAKE_jet', 'FE_PAKE'], ['P_EN_jet', 'FE_EN'], ['Area_jet', '1']]:
        out['rt/' + lum] = shell_sum(dump, flux, mask=is_jet)
    for lum, flux in [['Area_mag', '1']]:
        out['rt/' + lum] = shell_sum(dump, flux, mask=(dump['sigma'] > 1))
    for var in ['rho', 'Pg', 'u^r', 'u^th', 'u^3', 'b^r', 'b^th', 'b^3', 'b', 'inv_beta', 'Ptot']:
        out['rt/' + var + '_jet'] = shell_avg(dump, var, mask=is_jet)
    del is_jet
    

def lumproxy(dump, out, **kwargs):
    """Luminosity proxy from GRMHD code comparison '19
    """
    rho, Pg, B = dump['rho'], dump['Pg'], dump['b']
    # See EHT code comparison paper
    j = rho ** 3 / Pg ** 2 * np.exp(-0.2 * (rho ** 2 / (B * Pg ** 2)) ** (1. / 3.))
    out['rt/Lum'] = shell_sum(dump, j, j_slice=get_j_slice(dump))

def gridtotals(dump, out):
    """Total energy and current, summed by shells to allow any cut on radius
    """
    for tot_name, var_name in [['Etot', 'JE0'], ['Jsq_inv', 'jsq'], ['Jsq_loc', 'current']]:
        out['rt/'+tot_name] = shell_sum(dump, var_name)

def efluxes(dump, out, **kwargs):
    """Total energy fluxes, recorded so that ~div-free steady state flux can be computed
    """
    for var in ['JE0', 'JE1', 'JE2']:
        out['rt/'+var] = shell_sum(dump, var)
        if _get(kwargs, 'do_tavgs'):
            out['rth/' + var] = dump[var].mean(axis=-1)

# Total outflowing portions of variables
def outfluxes(dump, out, **kwargs):
    """Outflowing portions of fluxes
    """
    for name, var in [['outflow', 'FM'], ['outEflow', 'FE']]:
        var_tmp = dump[var]
        out['rt/'+name] = shell_sum(dump, var_tmp, mask=(var_tmp > 0))
        if _get(kwargs, 'do_tavgs'):
            out['r/'+name] = out['rt/'+name]

def pdfs(dump, out, **kwargs):
    """Probability density functions for certain variables. Even more likely than most analyses to be broken
    """
    for var, pdf_range in [ ['inv_beta', [-3.5, 3.5]], ['rho', [-7, 1]] ]:
        # TODO handle negatives, pass on the range & bins
        var_tmp = np.log10(dump[var])
        out['pdft/' + var], _ = np.histogram(var_tmp, bins=_get(kwargs, 'pdf_nbins'), range=pdf_range,
                                              weights=np.repeat(dump['gdet'], var_tmp.shape[2]).reshape(var_tmp.shape),
                                              density=True)
        del var_tmp

def omega_bz_advanced(dump, out, **kwargs):
    """A battery of different measurements of the Blandford-Znajek prediction of the B field rotation rate.
    """
    if _get(kwargs, 'do_tavgs'):
        Fcov01, Fcov13 = F_cov(dump, 0, 1), F_cov(dump, 1, 3)
        Fcov02, Fcov23 = F_cov(dump, 0, 2), F_cov(dump, 2, 3)
        vr, vth, vphi = dump['u^1']/dump['u^0'], dump['u^2']/dump['u^0'], dump['u^3']/dump['u^0']
        out['rhth/omega'] = np.zeros((dump['n1'],dump['n2']//2))
        out['rhth/omega_alt_num'] = np.zeros((dump['n1'],dump['n2']//2))
        out['rhth/omega_alt_den'] = np.zeros((dump['n1'],dump['n2']//2))
        out['rhth/omega_alt'] = np.zeros((dump['n1'],dump['n2']//2))
        out['rhth/vphi'] = np.zeros((dump['n1'],dump['n2']//2))
        out['rhth/F13'] = np.zeros((dump['n1'],dump['n2']//2))
        out['rhth/F01'] = np.zeros((dump['n1'],dump['n2']//2))
        out['rhth/F23'] = np.zeros((dump['n1'],dump['n2']//2))
        out['rhth/F02'] = np.zeros((dump['n1'],dump['n2']//2))
        coord_hth = dump.grid.coord_all()[:,:,:dump['n2']//2,0]
        alpha_over_omega =  dump['lapse'][:, :dump['n2']//2] / (dump['r_eh'] * np.sin(dump.grid.coords.th(coord_hth)))
        for i in range(dump['n1']):
            out['rhth/F01'][i] = theta_profile(dump, Fcov01, i, 1)
            out['rhth/F13'][i] = theta_profile(dump, Fcov13, i, 1)
            out['rhth/F02'][i] = theta_profile(dump, Fcov02, i, 1)
            out['rhth/F23'][i] = theta_profile(dump, Fcov23, i, 1)
            out['rhth/omega'][i] =  out['rhth/F01'][i] / out['rhth/F13'][i]
            out['rhth/omega_alt_num'][i] = theta_profile(dump, vr * dump['B3']*dump['B2'] + vth * dump['B3']*dump['B1'], i, 1)
            out['rhth/omega_alt_den'][i] = theta_profile(dump, dump['B2']*dump['B1'], i, 1)
            out['rhth/omega_alt'][i] = theta_profile(dump, vr * dump['B3']/dump['B1'] + vth * dump['B3']/dump['B2'], i, 1)
            out['rhth/vphi'][i] = theta_profile(dump, vphi, i, 1)

        out['rhth/omega_alt'] *= -alpha_over_omega

    del Fcov01, Fcov13, vr, vth, vphi

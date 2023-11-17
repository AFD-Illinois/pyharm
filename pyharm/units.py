__license__ = """
 File: units.py
 
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

__doc__ = \
"""CGS units: constants, plus tools for scaling code units into CGS.
"""

import numpy as np

cgs = {
    'CL': 2.99792458e10,
    'QE': 4.80320680e-10,
    'EE': 4.80320680e-10,
    'ME': 9.1093826e-28,
    'MP': 1.67262171e-24,
    'MN': 1.67492728e-24,
    'HPL': 6.6260693e-27,
    'HBAR': 1.0545717e-27,
    'KBOL': 1.3806505e-16,
    'GNEWT': 6.6742e-8,
    'SIG': 5.670400e-5,
    'AR': 7.5657e-15,
    'THOMSON': 0.665245873e-24,
    'JY': 1.e-23,
    'PC': 3.085678e18,
    'AU': 1.49597870691e13,
    'MSOLAR': 1.989e33,
    'RSOLAR': 6.96e10,
    'LSOLAR': 3.827e33
}

def get_cgs():
    """Get just the list of constants in CGS.
    """
    return cgs


def get_units_M87(M_unit, tp_over_te=3):
    """Get units dict for MBH=6.2e9, i.e. M87.
    See get_units for details.
    """
    return get_units(6.2e9, M_unit, tp_over_te)

def get_units_SgrA(M_unit, tp_over_te=3):
    """Get units dict for MBH=6.2e9, i.e. M87.
    See get_units for details.
    """
    return get_units(4.14e6, M_unit, tp_over_te)

def get_units(MBH, M_unit, tp_over_te=3, gam=4/3):
    """Get derived units and certain quantities for a system, given a BH mass in Msolar,
    and accretion density M_unit.
    Arguments tp_over_te and gam only matter for calculating Thetae_unit.
    Also note the calculation of Mdotedd assumes 10% efficiency.

    :param MBH: Black hole mass in solar masses
    :param M_unit: Density unit in grams, as fit by imaging with e.g. ``ipole``
    """
    out = {}
    MBH *= cgs['MSOLAR'] # Take input in solar masses
    out['MBH'] = MBH
    out['M_unit'] = M_unit
    out['L_unit'] = L_unit = cgs['GNEWT']*MBH / cgs['CL']**2
    out['T_unit'] = L_unit / cgs['CL']

    out['RHO_unit'] = RHO_unit  = M_unit / (L_unit ** 3)
    out['U_unit'] = RHO_unit * cgs['CL'] ** 2
    out['B_unit'] = cgs['CL'] * np.sqrt(4. * np.pi * RHO_unit)
    out['Ne_unit'] = RHO_unit / (cgs['MP'] + cgs['ME'])

    if tp_over_te is not None:
        out['Thetae_unit'] = (gam - 1.) * cgs['MP'] / cgs['ME'] / (1. + tp_over_te)
    else:
        out['Thetae_unit'] = cgs['MP'] / cgs['ME']

    out['Mdotedd'] = 4.*np.pi * cgs['GNEWT'] * MBH * cgs['MP'] / (0.1 * cgs['CL'] * cgs['THOMSON'])

    # Add constants
    out.update(cgs)

    return out

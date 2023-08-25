__license__ = """
 File: pretty.py
 
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
"""This file provides a function 'pretty' which takes a variable name used in pyharm,
and returns the LaTeX form of the name, suitable for plot axes/titles.
"""

# TODO: optionally add code or CGS units to any name
# TODO: better tracking/extensibility

pretty_dict = {'rho': r"\rho",
            'dP': r"\Delta P",
            'bsq': r"b^{2}",
            'sigma': r"\sigma",
            'u': r"u",
            'u_t': r"u_{t}",
            'u^t': r"u^{t}",
            'u_r': r"u_{r}",
            'u^r': r"u^{r}",
            'u_th': r"u_{\theta}",
            'u^th': r"u^{\theta}",
            'u_phi': r"u_{\phi}",
            'u^phi': r"u^{\phi}",
            'b_r': r"b_{r}",
            'b^r': r"b^{r}",
            'b_th': r"b_{\theta}",
            'b^th': r"b^{\theta}",
            'b_phi': r"b_{\phi}",
            'b^phi': r"b^{\phi}",
            'FM': r"\mathrm{Number\;Flux}\;FM",
            'FE':r"\mathrm{Energy\;Flux}\;FE_{\mathrm{tot}}",
            'FE_EM': r"\mathrm{Electromagnetic\;Energy\;Flux}\;FE_{EM}",
            'FE_Fl': r"\mathrm{Fluid\;Energy\;Flux}\;FE_{Fl}",
            'FL':r"\mathrm{Angular\;Momentum\;Flux}\;FL_{\mathrm{tot}}",
            'FL_EM': r"FL_{\mathrm{EM}}",
            'FL_Fl': r"FL_{\mathrm{Fl}}",
            'Be_b': r"Be_{\mathrm{B}}",
            'Be_nob': r"Be_{\mathrm{Fluid}}",
            'Pg': r"P_g",
            'p': r"P_g",
            'Pb': r"P_b",
            'Ptot': r"P_{\mathrm{tot}}",
            'beta': r"\beta",
            'inv_beta': r"\beta^{-1}",
            'jcov': r"j_{\mu}",
            'jsq': r"j^{2}",
            'current': r"J^{2}",
            'B': r"B",
            'betagamma': r"\beta \gamma",
            'Theta': r"\Theta",
            'Thetap': r"\Theta_{\mathrm{e}}",
            'Thetae': r"\Theta_{\mathrm{p}}",
            'JE0': r"JE^{t}",
            'JE1': r"JE^{r}",
            'JE2': r"JE^{\theta}",
            'divB': r"\nabla \cdot B",
            'MaxDivB': r"\mathrm{max}\left(\nabla \cdot B \right)",
            # Results of reductions which are canonically named
            'MBH': r"M_{\mathrm{BH}}",
            'Mdot': r"\dot{M}",
            'mdot': r"\dot{M}",
            'Phi_b': r"\Phi_{BH}",
            'Edot': r"\dot{E}",
            'Ldot': r"\dot{L}",
            'phi_b': r"\frac{\Phi_{BH}}{\sqrt{\langle \dot{M} \rangle}}",
            'phi_b_per': r"\frac{\Phi_{BH}}{\sqrt{\dot{M}}}",
            'edot': r"\frac{\dot{E}}{\langle \dot{M} \rangle}",
            'edot_per': r"\frac{\dot{E}}{\dot{M}}",
            'ldot': r"\frac{\dot{L}}{\langle \dot{M} \rangle}",
            'ldot_per': r"\frac{\dot{L}}{\dot{M}}",
            'eff': r"\frac{\left| \dot{E} - \dot{M} \right|}{\langle \dot{M} \rangle}",
            'spinup': r"\frac{\dot{L} - 2 a \dot{E}}{\langle \dot{M} \rangle}",
            # Independent variables
            't': r"t \; \left( \frac{G M}{c^3} \right)",
            'x': r"x \; \left( \frac{G M}{c^2} \right)",
            'y': r"y \; \left( \frac{G M}{c^2} \right)",
            'z': r"z \; \left( \frac{G M}{c^2} \right)",
            'r': r"r \; \left( \frac{G M}{c^2} \right)",
            'th': r"\theta",
            'phi': r"\phi"
            }

def pretty(var, segment=False):
    """Return a pretty LaTeX form of the named variable"""

    pretty_var = ""
    # Strip any flags that don't result in a different string
    if "_post" in var:
        return pretty(var.replace("_post",""), segment=segment)
    if "_disk" in var:
        return pretty(var.replace("_disk",""), segment=segment) + " (disk-average)"

    # Break down the name and translate bits we know to Latex;
    # keeps anything we don't understand as-is, no formatting
    ret = var
    if var[:4] == "log_":
        ret = r"\log_{10} \left( "+pretty(var[4:], segment=True)+r" \right)"
    if var[:4] == "abs_":
        ret = r"\left| "+pretty(var[4:], segment=True)+r" \right|"
    if var[:4] == "neg_":
        ret = r"-" + pretty(var[4:], segment=True)
    if var[:4] == "inv_":
        ret = r"1 / " + pretty(var[4:], segment=True)
    if var in pretty_dict:
        ret = pretty_dict[var]
    
    if segment:
        return ret
    else:
        return r"$" + ret + r"$"
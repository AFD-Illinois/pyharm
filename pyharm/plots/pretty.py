
"""This file provides a function 'pretty' which takes a variable name used in pyharm,
and returns the LaTeX form of the name, suitable for plot axes/titles.
"""

# TODO: optionally add code or CGS units to any name

pretty_dict = {'rho': r"\rho",
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
            'FM': r"\mathrm{Number Flux} FM",
            'FE':r"\mathrm{Energy Flux} FE_{\mathrm{tot}}",
            'FE_EM': r"\mathrm{Electromagnetic Energy Flux} FE_{EM}",
            'FE_Fl': r"\mathrm{Fluid Energy Flux} FE_{Fl}",
            'FL':r"FL_{\mathrm{tot}}",
            'FL_EM': r"FL_{\mathrm{EM}}",
            'FL_Fl': r"FL_{\mathrm{Fl}}",
            'Be_b': r"Be_{\mathrm{B}}",
            'Be_nob': r"Be_{\mathrm{Fluid}}",
            'Pg': r"P_g",
            'p': r"P_g",
            'Pb': r"P_b",
            'Ptot': r"P_{\mathrm{tot}}",
            'beta': r"\beta",
            'betainv': r"\beta^{-1}",
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
            'phi_b': r"\Phi_{BH} / \sqrt{\dot{M}}",
            'edot': r"\dot{E} / \dot{M}",
            'ldot': r"\dot{L} / \dot{M}",
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
    if var[:4] == "log_":
        return r"$\log_{10} \left( "+pretty(var[4:], segment=True)+r" \right)$"
    if var[:4] == "abs_":
        if segment:
            return r"\left| "+pretty(var[4:], segment=True)+r" \right|"
        else:
            return r"$\left| "+pretty(var[4:], segment=True)+r" \right|$"
    elif var in pretty_dict:
        if segment:
            return pretty_dict[var]
        else:
            return r"$"+pretty_dict[var]+r"$"
    else:
        # Give up
        return var
__license__ = """
 File: interface.py
 
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

class DumpFile(object):
    """This interface provides the list of functions to implement for new file filters for pyharm.
    A good base set of variables would be to provide something logical for the HARM primitive variables
    RHO, UU, U1, U2, U3, B1, B2, B3. The provided index_of() function returns the expected indices 1-8 for
    these variables -- e.g. from most HARM-like output, read_var returns prims[:,:,:,index_of(var)] in most cases.

    The constructor must initialize a dictionary member self.params, containing at least enough members to initialize
    a Grid object (see Grid constructor docstring), as well as any single-scalar properties which the analysis or plotting
    code will need to access (e.g. fluid gamma, BH spin, times & timesteps, etc).
    Usually this is done via a member function self.read_params, which may be called on its own in future
    """

    prim_names_harm = ("RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3", "q", "dP")
    prim_names_kharma = ("rho", "u", "u1", "u2", "u3", "q", "dP")

    @classmethod
    def index_of(cls, vname, eprim_names=None, eprim_indices=None):
        vname = vname.replace("prims.","").replace("c.c.bulk.","")
        # This is provided in the interface, as a bunch of codes (iharmXd, Babel-converted KORAL & BHAC, etc)
        # use the same ordering for the first 8 variables
        # If you need to read more than those 8, override this method.
        if vname in cls.prim_names_harm:
            return cls.prim_names_harm.index(vname)
        elif vname in cls.prim_names_kharma:
            return cls.prim_names_kharma.index(vname)
        # Vectors
        elif vname == "uvec":
            return slice(cls.index_of("u1"), cls.index_of("u3")+1)
        elif vname == "B" or vname == "Bvec":
            return slice(cls.index_of("B1"), cls.index_of("B3")+1)
        # All
        elif vname == "prims" or vname == "primitives" or vname == "all":
            return slice(None)
        # TODO pflag?
        else:
            return None

    @classmethod
    def get_dump_time(cls, fname):
        """Quickly get just the simulation time represented in the dump file.
        For cutting on time without loading everything.
        """
        raise NotImplementedError

    def read_var(self, var, slice=None):
        """Read a variable 'var' from the file as a numpy array.
        Optionally read ghost zones if specified, or read only a slice 'slice' of the values from the file.
        Returns 'None' if the variable doesn't exist in the file.
        """
        raise NotImplementedError

#!/usr/bin/env python3 

# Example of generating a FluidState corresponding to Bondi accretion,
# then writing it out in the iharm3d/Illinois format.
# Useful as a template for generating your own arbitrary fluid dumps

import numpy as np
import pyharm
from pyharm.grmhd.bondi import get_bondi_fluid_state

# This generates a grid.  If writing a problem for e.g. imaging, make sure
# this is high enough res but don't exhaust your memory.
# There are lots more options to this function, check grid.py
G = pyharm.grid.make_some_grid('mks', 192, 128, 128)

# This is the bit for actually writing out a state onto this grid.
# Basically this is just calling a set of functions at each point, and
# assigning the results to specially named arrays in a dict.
bondi_state = get_bondi_fluid_state(1, 8, 5./3, G)

# Hopefully this should just work if you've set all the right keys.
# It tries to guess some parameters, and will warn if it's omitting anything
# technically part of the standard
# (https://github.com/AFD-Illinois/docs/wiki/GRMHD-Output-Format)
# if you need it to write something, make sure it's in `state.params`
pyharm.io.iharm3d.write_dump(bondi_state, "bondi.h5", np.float32)

#!/usr/bin/env python3

# Experimental, incomplete re-implementation of HARM with Python/Loopy

import os
import sys
import logging
import importlib
from datetime import datetime
import cProfile
import pstats

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# Local imports
# TODO standardize import scheme; related to learning setuptools/pythonpath standards
import pyHARM.parameters as parameters
import pyHARM.diag as diag
import pyHARM.h5io as h5io
from pyHARM.grid import Grid
from pyHARM.step import step

from pyHARM.debug_tools import plot_var

# Start logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Gather parameters from different sources in order
# Later sources override earlier
params = parameters.defaults()
params = parameters.parse_argv(params, sys.argv)
params = parameters.parse_dat(params, params['paramfile'])
params = parameters.override_from_argv(params, sys.argv)
params = parameters.fix(params)

# Setup an OpenCL context and queue for the default device
# This prompts the user if we're in an interactive terminal
params['ctx'] = cl.create_some_context()
print(params['ctx'])
params['queue'] = cl.CommandQueue(params['ctx'])

# Import correct problem file/initialization
sys.path.insert(1, os.path.join(sys.path[0], '../prob'))
problem = importlib.import_module(params['prob'] + "." + params['prob'])

# Arrange file paths
dump_path = os.path.join(params['outdir'], "dumps")
restart_path = os.path.join(params['outdir'], "restarts")
if not os.path.exists(dump_path):
    os.mkdir(dump_path)
if not os.path.exists(restart_path):
    os.mkdir(restart_path)

walltime_start = datetime.now()

restart_fname = os.path.join(restart_path, "restart.last")
if os.path.exists(restart_fname) and 0:
    pass  # TODO implement checkpointing properly someday
    #P, dt, tlog, ndump, ncheckpoint = read_checkpoint(params, restart_fname)
    #tdump = (ndump + 1) * params['dump_cadence']
    #tcheckpoint = (ndump + 1) * params['checkpoint_cadence']
else:
    G = Grid(params)
    logger.info("Grid generated: {}, size {} in {}".format(params['coordinates'], G.N,
                                                           (datetime.now() - walltime_start)))
    # Honestly gridfiles are probably not necessary now, but they're good sanity checks
    h5io.dump_grid(G, fname=os.path.join(dump_path, "grid.h5"))

    # Create an ndarray to hold initial values temporarily
    P = np.zeros(G.shapes.grid_primitives)
    # Initialize the problem
    problem.init(params, G, P)
    P = cl_array.to_device(params['queue'], P)

    # Initialize main variables
    print("tf = {}".format(params['tf']))
    t = 0.0
    nstep = 0

    dt = params['dt_start']
    tdump = params['dump_cadence']
    tlog = params['log_cadence']
    #tcheckpoint = params['checkpoint_cadence']

    h5io.dump(params, G, P.get(), t, dt, "dumps/dump_00000000.h5")
    ndump = 1

    #write_checkpoint(params, P, "restarts/restart_00000000.h5")
    #ncheckpoint = 1

if 'profile' in params and params['profile']:
    profiler = cProfile.Profile()

walltimes_log = []
nsteps_log = []
gridsize = G.GN[1]*G.GN[2]*G.GN[3]
while t < params['tf']:
    logger.info("t = {}; dt = {}; n = {}".format(t, dt, nstep))
    t += dt
    nstep += 1

    if 'profile' in params and params['profile'] and nstep > 1:
        profiler.enable()

    P, dt = step(params, G, P, dt)

    if 'profile' in params and params['profile'] and nstep > 1:
        profiler.disable()
        if nstep % 15 == 0:
            pstats.Stats(profiler).sort_stats('tottime').print_stats()

    if 'plot' in params and params['plot']:
        plot_var(G, P.get()[G.slices.RHO], "RHO")

    if t < params['tf']:
        if t >= tdump:
            if not isinstance(t, float) and not isinstance(t, np.ndarray):
                t = t.get()
            if not isinstance(dt, float) and not isinstance(dt, np.ndarray):
                dt = dt.get()
            h5io.dump(params, G, P.get(), t, dt, "dumps/dump_{:08d}.h5".format(ndump))
            ndump += 1
            tdump += params['dump_cadence']
        if t >= tlog:
            walltimes_log.append(datetime.now())
            nsteps_log.append(nstep)
            if len(walltimes_log) > 1:
                zcps_last = (nsteps_log[-1] - nsteps_log[-2]) * gridsize /\
                            (walltimes_log[-1] - walltimes_log[-2]).total_seconds()
            else:
                zcps_last = 0
            zcps_start = nstep * gridsize / (walltimes_log[-1] - walltime_start).total_seconds()
            logger.info("ZCPS since last log: {} since start: {}".format(zcps_last, zcps_start))
            logger.info(diag.get_log_line(G, P.get(), t))

            tlog += params['log_cadence']
        # if t >= tcheckpoint:
        #     write_checkpoint(params, P, os.path.join(restart_path, "restart_{:08d}.h5".format(ncheckpoint)))

logging.shutdown()

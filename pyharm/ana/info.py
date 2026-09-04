__license__ = """
 File: info.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020-2026, pyharm contributors
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

import os
import click
import io
import numpy as np
import glob
import h5py

from pathlib import Path

from astropy.io import ascii
from astropy.table import Table

import pyharm
# We're practically an io function
import pyharm.io as pio
import pyharm.io.logs_kharma as kio
from pyharm.parameters import parse_parthenon_dat

class SimulationRun(object):
    """Class for storing information about a simulation as a whole
    """
    # Name, dtype, alignment
    # dtypes: (t)ext, (f)loat, (e)xponential, (i)nt, (a)uto
    properties = {
        'run_name': ("Run name", "S32", "l"),
        'code': ("Simulation code", "S32", "l"),
        'version': ("Code version", "S32", "l"),
        'branch': ("Code branch", "S32", "l"),
        'code_SHA1': ("Code SHA1", "S32", "l"),
        'spherical': ("Spherical", "?", "c"),
        'resolution': ("Resolution", "S32", "c"),
        'coordinates': ("Coordinate sytem", "S32", "l"),
        'base': ("Base coordinate system", "S32", "l"),
        'transform': ("Coordinate transform", "S32", "l"),
        'nfiles':    ("# of dumps", "i4", "l"),
        'output_start_time': ("Start time", "f4", "l"),
        'output_end_time': ("End time", "f4", "l"),
    }
    derivations = {
        'resolution': lambda dump: f"{dump['n1']}x{dump['n2']}x{dump['n3']}"
    }

    def __init__(self, path, n_dumps_to_load=1, load_logs=True):
        """Construct a SimulationRun object from the run at path.
        Should be whatever path contains
        """
        if os.path.isdir(path):
            fnames = pio.get_fnames(path)
        else:
            fnames = [path,]

        self.data = {}
        first_folder = os.path.realpath(fnames[0]).split("/")[-2]
        if "dumps" in first_folder:
            self.data['folder_name'] = os.path.realpath(fnames[0]).split("/")[-3]
            #path = "/".join(os.path.realpath(fnames[0]).split("/")[:-3])
        else:
            self.data['folder_name'] = first_folder
            #path = "/".join(os.path.realpath(fnames[0]).split("/")[:-2])

        # TODO parse tag.tex if it's there
        self.data['run_name'] = self.data['folder_name']

        self.data['nfiles'] = len(fnames)
        self.data['code'] = pio.get_dump_type(fnames[0])
        self.data['output_start_time'] = pio.get_dump_time(fnames[0])
        self.data['output_end_time'] = pio.get_dump_time(fnames[-1])

        initial_dump = pyharm.load_dump(fnames[0])
        for key in self.properties.keys():
            if key in initial_dump.params:
                self.data[key] = initial_dump.params[key]
            elif key in self.derivations:
                self.data[key] = self.derivations[key](initial_dump)

        # Load SimulationDump of first dump to access more properties
        n_dump_to_load = 1
        self.dumps = [SimulationDump(fnames[i]) for i in range(n_dump_to_load)]
        # Load SimulationLogs for even more properties
        if load_logs:
            self.logs = SimulationLogs(path)
        else:
            self.logs = {}

        # TODO auto-add all of parameter file parameters

    @classmethod
    def table_names(cls, cols):
        names = []
        for col in cols:
            if col in cls.properties:
                names.append(cls.properties[col][0])
            elif col in SimulationDump.properties:
                names.append(SimulationDump.properties[col][0])
            elif col in SimulationLogs.properties:
                names.append(SimulationLogs.properties[col][0])
        return names

    @classmethod
    def table_types(cls, cols):
        types = []
        for col in cols:
            if col in cls.properties:
                types.append(cls.properties[col][1])
            elif col in SimulationDump.properties:
                types.append(SimulationDump.properties[col][1])
            elif col in SimulationLogs.properties:
                types.append(SimulationLogs.properties[col][1])
        return types

    @classmethod
    def table_alignments(cls, cols):
        alns = []
        for col in cols:
            if col in cls.properties:
                alns.append(cls.properties[col][2])
            elif col in SimulationDump.properties:
                alns.append(SimulationDump.properties[col][2])
            elif col in SimulationLogs.properties:
                alns.append(SimulationLogs.properties[col][1])
        return alns

    # Only this requires actual data
    def table_row(self, cols):
        row = []
        for col in cols:
            row.append(self[col])
        return row

    def table_mask(self, cols):
        masks = []
        for col in cols:
            if col in self.logs.properties and len(self.logs.properties[col]) > 3:
                masks.append(self.logs.properties[col][3](self.logs))
            else:
                masks.append(False)
        return masks

    def __str__(self):
        table = Table(names=("key", "Name", "value"), dtype=('S16', 'S32', 'S32'))
        for key in self.data.keys():
            table.add_row([key, self.properties[key][0], str(self.data[key])])
        ret = io.StringIO()
        ascii.write(table, ret, format='fixed_width_no_header')
        return ret.getvalue()

    def __getitem__(self, key):
        if key in self.properties:
            return self.data[key]
        elif key in self.dumps[0].properties:
            return self.dumps[0].calcs[key](self.dumps[0].dump)
        elif key in self.logs.properties:
            return self.logs.calcs[key](self.logs)
        else:
            raise KeyError

class SimulationLogs(object):
    """For retrieving and holding info from the simulation logs (history files and stdout)
    """
    properties = {
        'jobnum': ("Job #", "i4", "c"),
        'status_str': ("Status", "S32", "c"),
        'walltime': ("Walltime [h]", "f4", "c"),
        'simtime':  (r"Simtime [$t_g$]", "f4", "c"),
        'perf':     ("Performance [ZCPS]", "f4", "c"),
        'floors_raw': ("Floors", "i4", "c", lambda self: kio.job_flags(self.last_lines, "fflag") <= 0),
        'floors_pct': ("Floors %", "i4", "c", lambda self: kio.job_flags(self.last_lines, "fflag") <= 0),
        'pflags_raw':  ("PFlags", "i4", "c", lambda self: kio.job_flags(self.last_lines, "pflag") <= 0),
        'pflags_pct':  ("PFlags %", "i4", "c", lambda self: kio.job_flags(self.last_lines, "pflag") <= 0),
    }

    calcs = {
        # This is job # for current/most recent run
        'jobnum': lambda self: self.jobnum(self.loglist[-1].split('/')[-1]),
        'status': lambda self: kio.job_status(self.last_lines),
        'status_str': lambda self: kio.job_status(self.last_lines).name,
        'walltime': lambda self: kio.job_wall_time(self.last_lines),
        'simtime': lambda self: kio.job_sim_time(self.last_lines),
        'perf': lambda self: kio.job_perf(self.last_lines),
        'floors_raw': lambda self: kio.job_flags(self.last_lines, "fflag"),
        'floors_pct': lambda self: kio.job_flag_pct(self.last_lines, "fflag"),
        'pflags_raw': lambda self: kio.job_flags(self.last_lines, "pflag"),
        'pflags_pct': lambda self: kio.job_flag_pct(self.last_lines, "pflag"),
    }

    def __init__(self, path):
        realpath = os.path.realpath(path)
        self.loglist = glob.glob(os.path.join(realpath, "slurm-*.out"))
        # Skip empty dirs
        if len(self.loglist) == 0:
            raise ValueError(f"No log files at path: {realpath}!")

        self.loglist.sort(key=self.jobnum)

        # TODO arrange loading more
        self.last_lines = kio.read_stdout(self.loglist[-1], -100)

    # We can't rely on touch times or lexical order
    def jobnum(self, fname):
        return int(''.join(c for c in fname if c.isdigit()))

    def __str__(self):
        table = Table(names=("key", "Name", "value"), dtype=('S16', 'S32', 'S32'))
        for key in self.data.keys():
            table.add_row([key, self.properties[key][0], str(self.data[key])])
        ret = io.StringIO()
        ascii.write(table, ret, format='fixed_width_no_header')
        return ret.getvalue()


class SimulationResults(object):
    """For holding properties of the analysis results on a simulation
    """


class SimulationDump(object):
    """For holding basic properties (parameters and derived quantities) of a single dump file.
    NOT a substitute for `pyharm analysis`!
    """
    properties = {
        'time': ("Time", "f4", "c"),
        'rho_min': ("Min. rho", "f4", "c"),
        'rho_max': ("Max. rho", "f4", "c"),
        'u_min': ("Min. u", "f4", "c"),
        'u_max': ("Max. u", "f4", "c"),
        'sigma_min': ("Min. sigma", "f4", "c"),
        'sigma_max': ("Max. sigma", "f4", "c"),
        'beta_min_true': ("Min. beta (true)", "f4", "c"),
        'beta_max': ("Max. beta (true)", "f4", "c"),
        'beta_min': ("Min. beta (ratio of maxima)", "f4", "c"),
    }

    calcs = {
        'rho_min': lambda dump: np.min(dump['rho']),
        'rho_max': lambda dump: np.max(dump['rho']),
        'u_min': lambda dump: np.min(dump['u']),
        'u_max': lambda dump: np.max(dump['u']),
        'sigma_min': lambda dump: np.min(dump['sigma']),
        'sigma_max': lambda dump: np.max(dump['sigma']),
        # TODO(ratio of minima/maxima?)
        'beta_min_true': lambda dump: np.min(dump['beta']),
        'beta_max': lambda dump: np.max(dump['beta']),
        'beta_min': lambda dump: np.max(dump['Pg']) / np.max(dump['Pb']),
        'beta_avg': lambda dump: np.sum(dump['gdet']*dump['beta'])/np.sum(dump['gdet']*dump['1'])
    }

    basic_set = ('beta_min')
    mid_set = ('rho_min', 'rho_max', 'u_min', 'u_max', 'beta_min')

    def __init__(self, dump_name):
        """Construct a SimulationDump from a backing file with `dump_name`
        """
        self.name = os.path.basename(dump_name)
        self.dump = pyharm.load_dump(dump_name)

    def __del__(self):
        # Hopefully this is enough.  pyharm doesn't keep open handles
        del self.dump

    # TODO print_all that just prints everything

    def table_row(self, cols):
        row = []
        for col in cols:
            row.append(self.calcs[col](self.dump))
        return row

    def __str__(self):
        table = Table(names=("key", "Name", "Value"), dtype=('S16', 'S32', 'S32'))
        for key in self.mid_set:
            table.add_row([key, self.properties[key][0], str(self.calcs[key](self.dump))])
        ret = io.StringIO()
        ascii.write(table, ret, format='fixed_width_no_header')
        return ret.getvalue()

def print_hst_info(fname, verbose=False):
    log = kio.read_log(fname)
    print("KHARMA history file:")
    print("Keys:", log.keys())
    print("Log lines:", len(log['time']))
    print("Time range:", log['time'][0], log['time'][-1])

def print_dump_pars(fname):
    # TODO more flexible, not just full file contents
    with h5py.File(fname) as f:
        print(f['/Input'].attrs['File'])

def compare_dump_pars(fname1, fname2):
    """Function to compare the parameters between two dump files.
    Loads up dictionaries and compares them as sets.
    """
    # Load parameters from both paths
    pars1 = {}
    pars2 = {}
    
    # Try to load as dump files first
    try:
        dump1 = pyharm.load_dump(fname1)
        pars1 = dump1.get('parameters', {})
    except:
        # If not a dump file, try to read as a parameter file
        try:
            with open(fname1, 'r') as f:
                pars1 = parse_parthenon_dat(f.read())
        except:
            print(f"Could not parse {fname1} as dump or parameter file")
            exit(1)
    
    try:
        dump2 = pyharm.load_dump(fname2)
        pars2 = dump2.get('parameters', {})
    except:
        # If not a dump file, try to read as a parameter file
        try:
            with open(fname2, 'r') as f:
                pars2 = parse_parthenon_dat(f.read())
        except:
            print(f"Could not parse {fname2} as dump or parameter file")
            exit(1)
    
    # Compare parameters
    print("Comparing parameters from:")
    print(f"  {fname1}")
    print(f"  {fname2}")
    print()
    
    # Simple comparison function based on example
    def compare_dicts(dict1, dict2):
        set1 = set(dict1.items())
        set2 = set(dict2.items())
        diff = set1 ^ set2
        if len(diff) > 0:
            print("Differences found:")
            for item in diff:
                print(f"  {item}")
        else:
            print("No differences found")
    
    # Compare parameters
    compare_dicts(pars1, pars2)

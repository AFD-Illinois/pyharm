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
import pyharm.io as pio
import pyharm.io.logs_kharma as kio
from pyharm.parameters import parse_parthenon_dat

class SimulationRun(object):
    """Class for storing information about a simulation as a whole
    """
    # Name, dtype, alignment
    # dtypes: (t)ext, (f)loat, (e)xponential, (i)nt, (a)uto
    properties = {
        # Run parameters
        'run_name': ("", "S32", "l", None),
        'sim_name': ("Name", "S32", "l", None),
        'code': ("Simulation code", "S32", "l", None),
        'version': ("Code version", "S32", "l", None),
        'branch': ("Code branch", "S32", "l", None),
        'code_SHA1': ("Code SHA1", "S32", "l", None),
        'spherical': ("Spherical", "?", "r", None),
        'resolution': ("Resolution", "S32", "r", None),
        'coordinates': ("Coordinate sytem", "S32", "r", None),
        'base': ("Base coordinate system", "S32", "r", None),
        'transform': ("Coordinate transform", "S32", "r", None),
        'nfiles':    ("# of dumps", "i4", "r", None),
        'output_start_time': ("Start time", "f4", "r", None),
        'output_end_time': ("End time", "f4", "r", None),
    }
    derivations = {
        'resolution': lambda dump: f"{dump['n1']}x{dump['n2']}x{dump['n3']}"
    }

    def __init__(self, path, n_dumps_to_load=1, load_logs=True, ana_fname="", arange=None):
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

        tex_tag_path = os.path.join(os.path.realpath(path), "tag.tex")
        if os.path.isfile(tex_tag_path):
            with open(tex_tag_path) as f:
                self.data['run_name'] = f.readline()[:-1]
        else:
            self.data['run_name'] = self.data['folder_name']

        text_sim_path = os.path.join(os.path.realpath(path), "name.txt")
        if os.path.isfile(text_sim_path):
            with open(text_sim_path) as f:
                self.data['sim_name'] = r"\texttt{" + f.readline()[:-1] + r"}"
        else:
            self.data['sim_name'] = self.data['folder_name']

        self.data['nfiles'] = len(fnames)
        self.data['code'] = pio.get_dump_type(fnames[0])
        self.data['output_start_time'] = pio.get_dump_time(fnames[0])
        self.data['output_end_time'] = pio.get_dump_time(fnames[-1])

        try:
            initial_dump = pyharm.load_dump(fnames[0])
            for key in self.properties.keys():
                if key in initial_dump.params:
                    self.data[key] = initial_dump.params[key]
                elif key in self.derivations:
                    self.data[key] = self.derivations[key](initial_dump)

            # Load SimulationDump of first dump to access more properties
            n_dump_to_load = 1
            self.dumps = [SimulationDump(fnames[i]) for i in range(n_dump_to_load)]

        except Exception as e:
            print(e)

        # Load SimulationPrints for even more properties
        if load_logs:
            try:
                self.logs = SimulationPrints(path)
            except Exception as e:
                print(e)
                self.logs = {}
        else:
            self.logs = {}

        if ana_fname != "":
            try:
                self.ana = SimulationResults(os.path.join(path, ana_fname), arange=arange)
            except Exception as e:
                print(e)
                self.ana = SimulationResults()
        else:
            self.ana = SimulationResults()

        # TODO auto-add all parameter file parameters to properties

    def __str__(self):
        # TODO everything?
        table = Table(names=("key", "Name", "value"), dtype=('S16', 'S32', 'S32'))
        for key in self.data.keys():
            table.add_row([key, self.properties[key][0], str(self.data[key])])
        ret = io.StringIO()
        ascii.write(table, ret, format='fixed_width_no_header')
        return ret.getvalue()

    # Just getting properties agnostically
    @classmethod
    def table_names(cls, cols):
        return [cls.get_property(col)[0] for col in cols]
    @classmethod
    def table_types(cls, cols):
        return [cls.get_property(col)[1] for col in cols]
    @classmethod
    def table_alignments(cls, cols):
        return [cls.get_property(col)[2] for col in cols]
    @classmethod
    def set_table_formats(cls, table, cols):
        # TODO is this available at creation like the others?
        for col, name in zip(cols, cls.table_names(cols)):
            if cls.get_property(col)[3] is not None:
                table[name].info.format = cls.get_property(col)[3]
        return table

    # Only this requires actual data
    def table_row(self, cols):
        return [self[col] for col in cols]

    def table_mask(self, cols):
        return [self.get_mask(col) for col in cols]

    def __contains__(self, item):
        # TODO check these are actually loaded!
        if key in self.properties:
            return True
        elif key in self.dumps[0].properties:
            return True
        elif key in self.logs.properties:
            return True
        elif key in self.ana.properties:
            return True
        else:
            return False

    @classmethod
    def get_property(cls, key):
        if key in cls.properties:
            return cls.properties[key]
        elif key in SimulationDump.properties:
            return SimulationDump.properties[key]
        elif key in SimulationPrints.properties:
            return SimulationPrints.properties[key]
        elif key in SimulationResults.properties:
            return SimulationResults.properties[key]
        else:
            raise KeyError

    # This requires data to determine the mask values
    def get_mask(self, key):
        if len(self.get_property(key)) > 4:
            if key in self.properties:
                return self.properties[key][4](self)
            elif key in self.dumps[0].properties:
                return self.dumps[0].properties[key][4](self.dumps[0])
            elif key in self.logs.properties:
                return self.logs.properties[key][4](self.logs)
            elif key in self.ana.properties:
                return self.ana.properties[key][4](self.ana)
        else:
            return False

    def __getitem__(self, key):
        if key in self.properties:
            return self.data[key]
        elif key in self.dumps[0].properties:
            return self.dumps[0].calcs[key](self.dumps[0])
        elif key in self.logs.properties:
            return self.logs.calcs[key](self.logs)
        elif key in self.ana.properties:
            return self.ana.calcs[key](self.ana)
        else:
            raise KeyError

class SimulationPrints(object):
    """For retrieving and holding info from the simulation logs (history files and stdout)
    """
    properties = {
        'jobnum': ("Job #", "i4", "r", None),
        'status_str': ("Status", "S32", "r", None),
        'walltime': ("Walltime [h]", "f4", "r", None),
        'simtime':  (r"Simtime [$t_g$]", "f4", "r", None),
        'perf':     ("Performance [ZCPS]", "f4", "r", None),
        'floors_raw': ("Floors", "i4", "r", None, lambda self: kio.job_flags(self.last_lines, "fflag") <= 0),
        'floors_pct': ("Floors %", "i4", "r", None, lambda self: kio.job_flags(self.last_lines, "fflag") <= 0),
        'pflags_raw':  ("PFlags", "i4", "r", None, lambda self: kio.job_flags(self.last_lines, "pflag") <= 0),
        'pflags_pct':  ("PFlags %", "i4", "r", None, lambda self: kio.job_flags(self.last_lines, "pflag") <= 0),
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

class SimulationLog(object):
    """For holding properties of the simulation history file
    """
    properties = {
        'log_keys': ("Log columns", "S1024", "r", None),
        'nlines':   ("# log lines", "i4", "r", None),
        'log_start_time': ("Log start time", "f4", "r", None),
        'log_end_time':   ("Log end time", "f4", "r", None), 
    }
    calcs = {
        'log_keys': lambda self: self.log.keys(),
        'nlines': lambda self: len(self.log['time']),
        'log_start_time': lambda self: self.log['time'][0],
        'log_end_time': lambda self: self.log['time'][-1],
    }

    def __init__(self, fname):
        # This is likely a bad idea
        self.log = io.read_log(fname)
        self.log_results = AnaResults(fname)

class SimulationResults(object):
    """For holding properties of the analysis results on a simulation
    """
    properties = {
        'peak_phi_b': (r"Peak $\phi_B$", "f4", "r", ".5g", lambda self: not np.isfinite(self.op_arange('phi_b', np.max, None))),
        'peak_eta':   (r"Peak $\eta$", "f4", "r", ".5g", lambda self: not np.isfinite(self.op_arange('eff', np.max, None))),

        'avg_phi_b':  (r"Late $\phi_B$", "f4", "r", ".5g", lambda self: not np.isfinite(self.op_arange('phi_b', np.mean, self.arange))),
        'avg_eta':    (r"Late $\eta$", "f4", "r", ".5g", lambda self: not np.isfinite(self.op_arange('eff', np.mean, self.arange))),
    }

    calcs = {
        'peak_phi_b': lambda self: self.op_arange('phi_b', np.max, None),
        'peak_eta': lambda self: self.op_arange('eff', np.max, None),

        'avg_phi_b': lambda self: self.op_arange('phi_b', np.mean, self.arange),
        'avg_eta': lambda self: self.op_arange('eff', np.mean, self.arange),
    }

    def __init__(self, fname=None, arange=None):
        if fname is not None:
            # avg_ends here sets the normalization for single-average norms
            # TODO set with sane default
            self.result = pyharm.load_result(fname, avg_ends=arange)
            # Our (optional) arange for our averages
            self.arange = arange


    def op_arange(self, var, op, arange):
        data = self.result[f't/{var}']

        # TODO special-case somewhere else
        if 'phi_b' in var:
            data *= np.sqrt(4*np.pi)

        if self.result.prefer_hst:
            time = self.result['diag/time']
        else:
            time = self.result['t']

        if arange is not None:
            # Get the times to average
            avg_slice = self.result.get_time_slice(*arange)
            times = (round(time[avg_slice][0]/1000)*1000,
                    round(time[avg_slice][-1]/1000)*1000)
            return op(data[avg_slice])
        else:
            return op(data)


class SimulationDump(object):
    """For holding basic properties (parameters and derived quantities) of a single dump file.
    NOT a substitute for `pyharm analysis`!
    """
    properties = {
        'time': ("Time", "f4", "r", None),
        'rho_min': ("Min. rho", "f4", "r", None),
        'rho_max': ("Max. rho", "f4", "r", None),
        'u_min': ("Min. u", "f4", "r", None),
        'u_max': ("Max. u", "f4", "r", None),
        'sigma_min': ("Min. sigma", "f4", "r", None),
        'sigma_max': ("Max. sigma", "f4", "r", None),
        'beta_min_true': ("Min. beta (true)", "f4", "r", None),
        'beta_max': ("Max. beta (true)", "f4", "r", None),
        'beta_min': (r"$P_{g,{\rm max}}/P_{b,{\rm max}}$", "f4", "r", ".5g"),
    }

    calcs = {
        'rho_min': lambda self: np.min(self.dump['rho']),
        'rho_max': lambda self: np.max(self.dump['rho']),
        'u_min': lambda self: np.min(self.dump['u']),
        'u_max': lambda self: np.max(self.dump['u']),
        'sigma_min': lambda self: np.min(self.dump['sigma']),
        'sigma_max': lambda self: np.max(self.dump['sigma']),
        # TODO(ratio of minima/maxima?)
        'beta_min_true': lambda self: np.min(self.dump['beta']),
        'beta_max': lambda self: np.max(self.dump['beta']),
        'beta_min': lambda self: np.max(self.dump['Pg']) / np.max(self.dump['Pb']),
        'beta_avg': lambda self: np.sum(self.dump['gdet']*self.dump['beta'])/np.sum(self.dump['gdet']*self.dump['1'])
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



# GENERAL FUNCTIONS


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

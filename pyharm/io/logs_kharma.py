__license__ = """
 File: logs_kharma.py
 
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

import sys
import glob
import os
import re

from .. import parameters
from ..defs import Loci, JobReturn

def read_stdout(fname, nlines=None):
    """Reads stdout capture file from KHARMA (e.g., slurm-XXXXXX.out).
    Optionally read only X lines (negative reads backward from end)
    """
    with open(fname, 'rb') as f:
        if nlines == None:
            raise NotImplementedError("full file reads not implemented")
        elif nlines < 0:
            try:  # catch OSError in case of a one line file 
                f.seek(-2, os.SEEK_END)
                lines_read = 0
                # Scroll back several lines
                while lines_read <= -nlines:
                    if f.read(1) == b'\n':
                        lines_read += 1
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)

            lines = []
            for l in range(lines_read+1):
                lines.append(f.readline().decode())
            lines = lines[1:]
        elif nlines > 0:
            raise NotImplementedError("top-file reads not implemented")

    return lines

def job_status(lines):
    """Return a single status for a job (running, exited, failed, etc.) from output lines.
    Should only need last ~10 lines.
    """
    job_crashed = False
    for line in reversed(lines):
        if "Aborted" in line:
            job_crashed = True
        elif "Segmentation fault" in line:
            return JobReturn.SEGFAULT
        elif "DUE TO TIME LIMIT" in line:
            return JobReturn.TIMELIMIT
        elif "zone-cycles/wallsecond" in line:
            return JobReturn.COMPLETED
        elif "DUE to SIGNAL Terminated" in line:
            return JobReturn.KILLED
        elif "major: File accessibility" in line:
            return JobReturn.CRASHED_FILE

    # If these phrases aren't in last X lines, script is probably (?) still running
    if job_crashed:
        return JobReturn.CRASHED
    else:
        return JobReturn.RUNNING

def job_sim_time(lines):
    """Return simulation time based on output lines.
    """
    return job_stat(lines, "time")

def job_wall_time(lines):
    """Return total wallclock time a run has used, based on output lines.
    """
    return job_stat(lines, "wsec_total") / 3600.

def job_perf(lines):
    """Return simulation current performance based on output lines
    """
    return job_stat(lines, "zone-cycles/wsec_step")

def job_stat(lines, stat):
    """Return simulation time based on output lines.
    Should only need last ~10 lines.
    """
    for line in reversed(lines):
        if f"{stat}=" in line:
            for seg in line.split(" "):
                if f"{stat}=" in seg:
                    return float(seg.replace(f"{stat}=",""))
    return 0.

# TODO some other notfound?
def job_flags(lines, type="pflag"):
    """Return total number of inverter flags"""
    for line in reversed(lines):
        if f"{type}: " in line:
            return int(line.split(" ")[1])
    return -1

def job_flag_pct(lines, type="pflag"):
    """Return total number of inverter flags"""
    for line in reversed(lines):
        if f"{type}: " in line:
            return int(line.split(" ")[2].lstrip("(").rstrip("%"))
    return -1

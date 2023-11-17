__license__ = """
 File: util.py
 
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
"""Generic functions which have no pyharm (or indeed any) dependencies.
Currently, mostly index handling.
"""

def slice_to_index(current_start, current_stop, slc):
    """Take a slice out of a range represented by start and end points.
    Resolves positive, negative, and integer slice values correctly.
    """
    new_start = list(current_start).copy()
    new_stop = list(current_stop).copy()
    for i in range(len(slc)):
        if isinstance(slc[i], int):
            new_start[i] = current_start[i] + slc[i]
            new_stop[i] = current_start[i] + slc[i] + 1
        elif slc[i] is not None:
            if slc[i].start is not None:
                new_start[i] = current_start[i] + slc[i].start if slc[i].start >= 0 else current_stop[i] + slc[i].stop
            if slc[i].stop is not None:
                # Count forward from global_start, or backward from global_stop
                new_stop[i] = current_start[i] + slc[i].stop if slc[i].stop >= 0 else current_stop[i] + slc[i].stop

    return new_start, new_stop

def i_of(var, val, behind=True, fail=False):
    """Convenience for finding zone containing a given value,
    in coordinate/monotonic-increase variables
    """
    i = 0
    while var[i] < val:
        i += 1
        # Warn or fail if we step too far
        if i == len(var):
            if fail:
                raise ValueError("Array does not contain value {}".format(val))
            else:
                print("Warning: using last value {} as desired value {}".format(var[-1], val))
                break

    # Return zone before the value, usually what we want for fluxes
    if behind or i == len(var):
        i -= 1

    return i

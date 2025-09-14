import sys
import code

import yt

__doc__ = \
"""This opens a prompt with yt loaded, and optionally a dump file.
Can add other imports/environment as necessary.
"""

environment = dict(yt=yt)

# TODO use autodetection from what-is to load results, etc.?
if len(sys.argv) > 1:
    if sys.argv[1] == '--help' or sys.argv[1] == '-h':
        print("Usage: gryt prompt [dumpfile]\n")
        print("\tThis command activates a python prompt with yt loaded.")
        print("\tTo also load a dump file (as 'dump' in the prompt),")
        print("\tinclude the dump file name as the last argument.")
        exit()
    else:
        environment['dump'] = yt.load(sys.argv[1])

code.interact(local=environment)
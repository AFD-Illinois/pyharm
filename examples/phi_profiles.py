#!/usr/bin/env python3

import sys
import numpy as np
import pyharm
from pyharm.ana import reductions
from matplotlib import pyplot as plt

fig, ax = plt.subplots(1,1, figsize=(10,10))

for dumpname in sys.argv[1:]:
    dump = pyharm.load_dump(dumpname)
    phi_at = 0.5 * reductions.shell_sum(dump, 'abs_B1')
    plt.plot(dump['r1d'], phi_at, label=dumpname)

plt.xlim(0, 50.)
plt.ylim(0., 100.)
plt.legend()
plt.grid()

fig.savefig('phi_profiles.png')


import sys
import h5py

outf = h5py.File("average.h5", "w")

fnames = sys.argv[1:]
nfiles = len(fnames)

header_copied = False
for fname in fnames:
    inf = h5py.File(fname, "r")
    if not header_copied:
        #outf['header'] = 
        inf['header'].copy(inf, outf, name='header')
        outf['prims'] = inf['prims'][()]/nfiles
        header_copied = True
    else:
        outf['prims'][()] += inf['prims'][()]/nfiles
    inf.close()

outf.close()
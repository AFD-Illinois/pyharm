## xdmf_output.py: generate an xdmf file for an output dump

import sys
from pyHARM import load_dump
from pathlib import Path

dumpname = sys.argv[1]
dump = load_dump(dumpname, add_derived=False, add_fail=False)
xmfpath = dumpname[:-16]+"xmf/"
xmfname = xmfpath + "dump_%08d.xmf" % dump['n_dump']
print("Writing {}".format(xmfname))
Path(xmfpath).mkdir(parents=True, exist_ok=True)

N1TOT = dump['n1']
N2TOT = dump['n2']
N3TOT = dump['n3']
NDIM = dump['n_dim']
NVAR = dump['n_prim']
t = dump['t']
vnams = dump['prim_names']

def geom_meta(fp):
    fp.write("      <!-- GRID DEFINITION -->\n")
    fp.write("      <Geometry GeometryType=\"X_Y_Z\">\n")
    for d in range(1,NDIM):
        coord_meta(fp, d)
    fp.write("      </Geometry>\n")

def vec_meta(fp, name):
    for mu in range(NDIM):
        vec_component(fp, name, mu)

def tensor_meta(fp, name, sourcename, precision):
    for mu in range(NDIM):
        for nu in range(NDIM):
            tensor_component(fp, name, sourcename, precision, mu, nu)

def coord_meta(fp, indx):
    fp.write(
        "        <DataItem ItemType=\"Hyperslab\" Dimensions=\"%d %d %d\" "
        "Type=\"Hyperslab\">\n" % (N1TOT + 1, N2TOT + 1, N3TOT + 1))
    fp.write("          <DataItem Dimensions=\"3 4\" Format=\"XML\">\n")
    fp.write("            0 0 0 %d\n" % indx)
    fp.write("            1 1 1 1\n")
    fp.write("            %d %d %d 1\n" % (N1TOT + 1, N2TOT + 1, N3TOT + 1))
    fp.write("          </DataItem>\n")
    fp.write(
        "          <DataItem Dimensions=\"%d %d %d %d\" NumberType=\"Float\" "
        "Precision=\"64\" Format=\"HDF\">\n" % (N1TOT + 1, N2TOT + 1, N3TOT + 1, NDIM))
    fp.write("            %s:/XFcart\n" % gname)
    fp.write("          </DataItem>\n")
    fp.write("        </DataItem>\n")

def prim_meta(fp, vnams, indx):
    fp.write(
        "      <Attribute Name=\"%s\" AttributeType=\"Scalar\" "
        "Center=\"Cell\">\n" % vnams[indx])
    fp.write(
        "        <DataItem ItemType=\"Hyperslab\" Dimensions=\"%d %d %d\" "
        "Type=\"Hyperslab\">\n" % (N1TOT, N2TOT, N3TOT))
    fp.write("          <DataItem Dimensions=\"3 4\" Format=\"XML\">\n")
    fp.write("            0 0 0 %d\n" % indx)
    fp.write("            1 1 1 1\n")
    fp.write("            %d %d %d 1\n" % (N1TOT, N2TOT, N3TOT))
    fp.write("          </DataItem>\n")
    fp.write(
        "          <DataItem Dimensions=\"%d %d %d %d\" NumberType=\"Float\" "
        "Precision=\"32\" Format=\"HDF\">\n" % (N1TOT, N2TOT, N3TOT, NVAR))
    fp.write("            %s:/prims\n" % dname)
    fp.write("          </DataItem>\n")
    fp.write("        </DataItem>\n")
    fp.write("      </Attribute>\n")

def scalar_meta(fp, name, sourcename, precision):
    fp.write(
        "      <Attribute Name=\"%s\" AttributeType=\"Scalar\" "
        "Center=\"Cell\">\n" % name)
    fp.write(
        "        <DataItem ItemType=\"Hyperslab\" Dimensions=\"%d %d %d\" "
        "Type=\"Hyperslab\">\n"% (N1TOT, N2TOT, N3TOT))
    fp.write("          <DataItem Dimensions=\"3 3\" Format=\"XML\">\n")
    fp.write("            0 0 0\n")
    fp.write("            1 1 1\n")
    fp.write("            %d %d %d\n" % (N1TOT, N2TOT, N3TOT))
    fp.write("          </DataItem>\n")
    fp.write(
        "          <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
        "Precision=\"%d\" Format=\"HDF\">\n" % (N1TOT, N2TOT, N3TOT, precision))
    fp.write("            %s:/%s\n" % (sourcename, name))
    fp.write("          </DataItem>\n")
    fp.write("        </DataItem>\n")
    fp.write("      </Attribute>\n")

def vec_component(fp, name, indx):
    fp.write(
        "      <Attribute Name=\"%s_%d\" AttributeType=\"Scalar\" "
        "Center=\"Cell\">\n" % (name, indx))
    fp.write(
        "        <DataItem ItemType=\"Hyperslab\" Dimensions=\"%d %d %d\" "
        "Type=\"Hyperslab\">\n" % (N1TOT, N2TOT, N3TOT))
    fp.write("          <DataItem Dimensions=\"3 4\" Format=\"XML\">\n")
    fp.write("            0 0 0 %d\n" % indx)
    fp.write("            1 1 1 1 \n")
    fp.write("            %d %d %d 1\n" % (N1TOT, N2TOT, N3TOT))
    fp.write("          </DataItem>\n")
    fp.write(
        "          <DataItem Dimensions=\"%d %d %d %d\" NumberType=\"Float\" "
        "Precision=\"32\" Format=\"HDF\">\n" % (N1TOT, N2TOT, N3TOT, NDIM))
    fp.write("            %s:/%s\n" % (fname, name))
    fp.write("          </DataItem>\n")
    fp.write("        </DataItem>\n")
    fp.write("      </Attribute>\n")

def tensor_component(fp, name, sourcename, precision, mu, nu):
    fp.write(
        "      <Attribute Name=\"%s_%d%d\" AttributeType=\"Scalar\" "
        "Center=\"Cell\">\n" % (name, mu, nu))
    fp.write(
        "        <DataItem ItemType=\"Hyperslab\" Dimensions=\"%d %d %d\" "
        "Type=\"Hyperslab\">\n" % (N1TOT, N2TOT, N3TOT))
    fp.write("          <DataItem Dimensions=\"3 5\" Format=\"XML\">\n")
    fp.write("            0 0 0 %d %d\n" % (mu, nu))
    fp.write("            1 1 1 1 1\n")
    fp.write("            %d %d %d 1 1\n" % (N1TOT, N2TOT, N3TOT))
    fp.write("          </DataItem>\n")
    fp.write(
        "          <DataItem Dimensions=\"%d %d %d %d %d\" NumberType=\"Float\" "
        "Precision=\"%d\" Format=\"HDF\">\n" % (N1TOT, N2TOT, N3TOT, NDIM, NDIM, precision))
    fp.write("            %s:/%s\n" % (sourcename, name))
    fp.write("          </DataItem>\n")
    fp.write("        </DataItem>\n")
    fp.write("      </Attribute>\n")

if __name__ == "__main__":
    dname = "../dump_%08d.h5" % dump['n_dump']
    gname = "../grid.h5"
    fname = "dump_%08d.xmf" % dump['n_dump']
    name = xmfname
    full_dump = dump['is_full_dump']

    fp = open(name, "w")

    # header
    fp.write("<?xml version=\"1.0\" ?>\n")
    fp.write("<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n")
    fp.write("<Xdmf Version=\"3.0\">\n")
    fp.write("  <Domain>\n")
    fp.write("    <Grid Name=\"mesh\" GridType=\"Uniform\">\n")
    fp.write("      <Time Value=\"%16.14e\"/>\n" % t)
    fp.write(
        "      <Topology TopologyType=\"3DSMesh\" NumberOfElements=\"%d %d "
        "%d\"/>\n" % (N1TOT + 1, N2TOT + 1, N3TOT + 1))

    geom_meta(fp) # Geometry
    fp.write("\n")

    # Jacobians
    # fp.write("      <!-- JACOBIANS -->\n")
    # fp.write("      <!-- contravariant -->\n")
    # tensor_meta(fp, "Lambda_h2cart_con", gname, 64)
    # fp.write("      <!-- covariant -->\n")
    # tensor_meta(fp, "Lambda_h2cart_cov", gname, 64)
    # fp.write("\n")

    # Metric
    fp.write("      <!-- METRIC -->\n")
    fp.write("      <!-- contravariant -->\n")
    tensor_meta(fp, "gcon", gname, 64)
    fp.write("      <!-- covariant -->\n")
    tensor_meta(fp, "gcov", gname, 64)
    fp.write("      <!-- determinant -->\n")
    scalar_meta(fp, "gdet", gname, 64)
    fp.write("      <!-- lapse -->\n")
    scalar_meta(fp, "alpha", gname, 64)
    fp.write("\n")

    # Variables
    fp.write("      <!-- PRIMITIVES -->\n")
    for p in range(dump['n_prim']):
        prim_meta(fp, vnams, p)
    fp.write("\n")

    if (full_dump):
        fp.write("      <!-- DERIVED VARS -->\n")
        # scalar_meta(fp, "divb", dname, 32)
        fp.write("      <!-- jcon -->\n")
        vec_meta(fp, "jcon")
        # ???
        # scalar_meta(fp, "PRESS", dname, 32)
        # scalar_meta(fp, "ENT", dname, 32)
        # scalar_meta(fp, "TEMP", dname, 32)
        # scalar_meta(fp, "CS2", dname, 32)


    # footer
    fp.write("    </Grid>\n")
    fp.write("  </Domain>\n")
    fp.write("</Xdmf>\n")

    fp.close()
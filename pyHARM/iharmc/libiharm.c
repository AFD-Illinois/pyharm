#include <stdio.h>
#include <string.h>

#include "u2p.h"

#define PIJK(P,I,J,K,N1,N2,N3) (K+(J+(I+P*N1)*N2)*N3)
#define MUNUIJ(MU,NU,I,J,N1,N2) (J+(I+(NU+MU*4)*N1)*N2)

void u2p (double prims[], double cons[], double gam, int eflag[],
          double gcov[], double gcon[], double gdet[], double lapse[],
          int NPRIM, int N1, int N2, int N3, int NG) {

  int N1G = N1 + 2*NG;
  int N2G = N2 + 2*NG;
  int N3G = N3 + 2*NG;

#pragma omp parallel for collapse(3)
  for (int i=NG; i<N1+NG; ++i) {
    for (int j=NG; j<N2+NG; ++j) {
      for (int k=NG; k<N3+NG; ++k) {

        // Grab elements p,i,j,k -> p
        double prims_here[NPRIM], cons_here[NPRIM];
        for (int p=0; p<NPRIM; ++p) {
          prims_here[p] = prims[PIJK(p,i,j,k,N1G,N2G,N3G)];
          cons_here[p] = cons[PIJK(p,i,j,k,N1G,N2G,N3G)];
        }

        double gcov_here[NDIM][NDIM], gcon_here[NDIM][NDIM];
        for (int mu=0; mu<NDIM; ++mu) {
          for (int nu=0; nu<NDIM; ++nu) {
            gcon_here[mu][nu] = gcon[MUNUIJ(mu,nu,i,j,N1G,N2G)];
            gcov_here[mu][nu] = gcov[MUNUIJ(mu,nu,i,j,N1G,N2G)];
          }
        }

        int zidx = PIJK(0,i,j,0,N1G,N2G,1);
        int eidx = PIJK(0,i,j,k,N1G,N2G,N3G);

        eflag[eidx] = U_to_P(gdet[zidx], lapse[zidx], prims_here, cons_here, gam, gcov_here, gcon_here);

        // Put the prims back in the prims
        for (int p=0; p<NPRIM; ++p) {
          prims[PIJK(p,i,j,k,N1G,N2G,N3G)] = prims_here[p];
        }

        //if (eflag[eidx] != 0) {
        //  fprintf(stderr, "%d %d %d -> %d\n", i,j,k, eflag[eidx]);
        //}

      }
    }
  }

}

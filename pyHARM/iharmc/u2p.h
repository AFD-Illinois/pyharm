#ifndef U2P_H
#define U2P_H

#include <math.h>

#define NDIM      (4)
#define ERRTOL    (1.e-8)
#define ITERMAX   (8)
#define DEL       (1.e-5)
#define GAMMAMAX  (50.)

#define P4VEC(S,X) fprintf(stderr, "%s: %g %g %g %g\n", S, X[0], X[1], X[2], X[3]);

double Pressure_rho0_u(double rho0, double u, double gam);
double Pressure_rho0_w(double rho0, double w, double gam);
double err_eqn(double Bsq, double D, double Ep, double QdB, double Qtsq, double Wp, double gam, int *eflag);
double gamma_func(double Bsq, double D, double QdB, double Qtsq, double Wp, int *eflag);
double Wp_func(double rho0, double u, double U1, double U2, double U3, double gam, double gcov[NDIM][NDIM], int *eflag);

int U_to_P(double gdet, double lapse, double prims[8], double cons[8], double gam, double gcov[NDIM][NDIM], double gcon[NDIM][NDIM]);

#endif // U2P_H
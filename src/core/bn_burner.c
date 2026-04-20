#include "bn_burner.h"

#include "store.h"

// #define __DIAG_ON
#include "diagnostic.h" // Bad header name... clashes with libc standards...?

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ZERO_FIX 1e-50
#define THIRD 1.0 / 3.0

// So that my LSP stops bugging me.
#ifndef SIZE
#define SIZE 16
#endif

#if SIZE == 16
#define SIZE 16
#define NUM_REACTIONS 48
#define NUM_FLUXES_PLUS 72
#define NUM_FLUXES_MINUS 72
#endif

#if SIZE == 150
#define SIZE 150
#define NUM_REACTIONS 1604
#define NUM_FLUXES_PLUS 2710
#define NUM_FLUXES_MINUS 2704
#endif

#if SIZE == 365
#define SIZE 365
#define NUM_REACTIONS 4395
#define NUM_FLUXES_PLUS 7429
#define NUM_FLUXES_MINUS 7420
#endif

static void hyperion_burner_kernel(double* tstep, double* temp, double* dens,
                                   double* xin, double* restrict xout,
                                   double* sdotrate);

// Interpretation from Fortran call:
//
//   hyperion_burner(tstep, temp, dens, xin(:), xout(:), ..., size)
//
//   `size` refers to the number of cells
//
// I don't know what the other vars are...

void hyperion_burner_(double* tstep, double* temp, double* dens, double* xin,
                      double* restrict xout, double* sdotrate,
                      uchar* burned_zone, int* size) {

    for (int i = 0; i < *size; i++) {
        hyperion_burner_kernel(tstep, &temp[i], &dens[i], xin + (SIZE * i),
                               xout + (SIZE * i), &sdotrate[i]);
    }
}

static void hyperion_burner_kernel(double* tstep, double* temp, double* dens,
                                   double* xin, double* restrict xout,
                                   double* sdotrate) {


    // We use the  XOUT memory, but our computation is with abundance (y)
    // Yes, this is confusing.
    
    for (int i = 0; i < SIZE; i++) {
        xout[i] = xin[i] / aa[i];
    }

    __DIAG_HALT("xin->xout", "xout", xout, SIZE);

    double tmp = *temp / 1e9;
    double ddt_e = 0;

    double t93 = pow(tmp, THIRD);
    double t1 = 1 / tmp;
    double t2 = 1 / t93;
    double t3 = t93;
    double t4 = tmp;
    double t5 = t93 * t93 * t93 * t93 * t93;
    double t6 = log(tmp);

    double density[3] = {1.0, *dens, (*dens) * (*dens)};

    for (int i = 0; i < num_reactions; i++) {
        double prefactor_ = prefactor[i] * density[num_react_species[i] - 1];
        rate[i] =
            prefactor_ * exp(p_0[i] + t1 * p_1[i] + t2 * p_2[i] + t3 * p_3[i] +
                             t4 * p_4[i] + t5 * p_5[i] + t6 * p_6[i]);
    }

    __DIAG_HALT("rates", "rate", rate, num_reactions);

    double t = 1e-20;
    double dt = t * 1e-2;

    while (t < *tstep) {

        __DIAG_HALT("Timestep", "dt", &dt, 1);

        for (int i = 0; i < num_reactions; i++) {
            switch (num_react_species[i]) {
            case 1:
                flux[i] = rate[i] * xout[reactant_1[i]];
                break;
            case 2:
                flux[i] = rate[i] * xout[reactant_1[i]] * xout[reactant_2[i]];
                break;
            case 3:
                flux[i] = rate[i] * xout[reactant_1[i]] * xout[reactant_2[i]] *
                          xout[reactant_3[i]];
                break;
            }
        }
        __DIAG_HALT("fluxes", "flux", flux, num_reactions);

        for (int i = 0; i < f_plus_total; i++) {
            f_plus[i] = f_plus_factor[i] * flux[f_plus_map[i]];
        }
        __DIAG_HALT("F+", "f_plus", f_plus, f_plus_total);

        for (int i = 0; i < f_minus_total; i++) {
            f_minus[i] = f_minus_factor[i] * flux[f_minus_map[i]];
        }
        __DIAG_HALT("F+", "f_minus", f_minus, f_minus_total);

        for (int i = 0; i < num_species; i++) {
            // TODO: these dont change. We should determine them BEFORE
            // integration. ESPECIALY since it's the biggest time-cost. If
            // we do that, it may be possible to build a very optimized
            // routine for it.
            // f_*_max uses a leading sentinel ([0] = -1); the "upper bound
            // for species i" now lives at [i + 1], and the (i == 0) guard
            // that used to wrap the lower bound is gone.
            int min_plus = f_plus_max[i] + 1;
            int min_minus = f_minus_max[i] + 1;
            f_plus_sum[i] = 0.0;
            for (int j = min_plus; j <= f_plus_max[i + 1]; j++) {
                f_plus_sum[i] += f_plus[j];
            }
            f_minus_sum[i] = 0.0;
            for (int j = min_minus; j <= f_minus_max[i + 1]; j++) {
                f_minus_sum[i] += f_minus[j];
            }
        }
        __DIAG_HALT("F+ sum", "f_plus_sum", f_plus_sum, num_species);
        __DIAG_HALT("F- sum", "f_minus_sum", f_minus_sum, num_species);

        for (int i = 0; i < num_species; i++) {
            if (f_minus_sum[i] * dt > xout[i]) {
                // Asympotic estimate.
                xout[i] =
                    (xout[i] + f_plus_sum[i] * dt) /
                    (1.0 + ((f_minus_sum[i] / (xout[i] + ZERO_FIX)) * dt));

                // NOTE: FMA is slower here (I suspect due to the blocking
                // of the accumulator)
            } else {
                // Forward Euler update.
                xout[i] += (f_plus_sum[i] - f_minus_sum[i]) * dt;
            }
        }
        __DIAG_HALT("Y step", "xout", xout, SIZE);

        // TODO: this continuous renormalization seems necessary, but it's
        // really not great for performance...
        double normalfac = 0;
        for (int i = 0; i < num_species; i++) {
            xout[i] = xout[i] * aa[i];
            normalfac += xout[i];
        }

        normalfac = 1 / normalfac;
        __DIAG_HALT("Normalization", "normalfac", &normalfac, 1);
        for (int i = 0; i < num_species; i++) {
            xout[i] = xout[i] * normalfac;
            xout[i] = xout[i] / aa[i];
        }

        tmp = 0;
        for (int i = 0; i < num_reactions; i++) {
            tmp += flux[i] * q_value[i];
        }
        tmp *= dt;
        ddt_e += tmp * 9.5768e17; // Convert MeV/nucleon/s to erg/g/s

        t += dt;
        dt = 1e-4 * t;
    }

    for (int i = 0; i < SIZE; i++) {
        xout[i] = xout[i] * aa[i];
    }

    *sdotrate = ddt_e;

    return;
}

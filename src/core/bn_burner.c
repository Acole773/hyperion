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
    double final_flux_energy = 0;

    // We use the  XOUT memory, but our computation is with abundance (y)
    // Yes, this is confusing.
    
    for (int i = 0; i < SIZE; i++) {
        xout[i] = xin[i] / aa[i];
    }

    __DIAG_HALT("xin->xout", "xout", xout, SIZE);

    double tmp = *temp / 1e9;
    double t93 = pow(tmp, THIRD);
    double t1 = 1 / tmp;
    double t2 = 1 / t93;
    double t3 = t93;
    double t4 = tmp;
    double t5 = t93 * t93 * t93 * t93 * t93;
    double t6 = log(tmp);

    double density[3] = {1.0, *dens, (*dens) * (*dens)};

    for (int i = 0; i < num_reactions; i++) {
        int ns = num_react_species[i] - 1;
        if (ns < 0) {
            ns = 0;
        } else if (ns > 2) {
            ns = 2;
        }

        double prefactor_ = prefactor[i] * density[ns];
        rate[i] =
            prefactor_ * exp(p_0[i] + t1 * p_1[i] + t2 * p_2[i] + t3 * p_3[i] +
                             t4 * p_4[i] + t5 * p_5[i] + t6 * p_6[i]);
    }

    __DIAG_HALT("rates", "rate", rate, num_reactions);

    double t = 1e-20;
    double dt = t * 1e-2;
    double t_end = *tstep;

    while (1) {
        if (t >= t_end || !isfinite(t) || dt <= 0.0) {
            break;
        }

        __DIAG_HALT("Timestep", "dt", &dt, 1);

        for (int i = 0; i < num_reactions; i++) {
            double f_rate = rate[i];
            double f = f_rate;

            if (num_react_species[i] > 0) {
                f *= xout[reactant_1[i]];
            }
            if (num_react_species[i] > 1) {
                f *= xout[reactant_2[i]];
            }
            if (num_react_species[i] > 2) {
                f *= xout[reactant_3[i]];
            }

            flux[i] = f;
        }
        __DIAG_HALT("fluxes", "flux", flux, num_reactions);

        for (int i = 0; i < num_species; i++) {
            int min_plus = (i == 0) ? 0 : f_plus_max[i - 1] + 1;
            int min_minus = (i == 0) ? 0 : f_minus_max[i - 1] + 1;
            double plus = 0.0;
            double minus = 0.0;

            for (int j = min_plus; j <= f_plus_max[i]; j++) {
                plus += f_plus_factor[j] * flux[f_plus_map[j]];
            }
            for (int j = min_minus; j <= f_minus_max[i]; j++) {
                minus += f_minus_factor[j] * flux[f_minus_map[j]];
            }

            f_plus_sum[i] = plus;
            f_minus_sum[i] = minus;

            if (minus * dt > xout[i]) {
                // Asympotic estimate.
                xout[i] =
                    (xout[i] + plus * dt) /
                    (1.0 + ((minus / (xout[i] + ZERO_FIX)) * dt));

                // NOTE: FMA is slower here (I suspect due to the blocking
                // of the accumulator)
            } else {
                // Forward Euler update.
                xout[i] += (plus - minus) * dt;
            }
        }
        __DIAG_HALT("F+ sum", "f_plus_sum", f_plus_sum, num_species);
        __DIAG_HALT("F- sum", "f_minus_sum", f_minus_sum, num_species);
        __DIAG_HALT("Y step", "xout", xout, SIZE);

        for (int i = 0; i < num_species; i++) {
            if (xout[i] < ZERO_FIX) {
                xout[i] = ZERO_FIX;
            }
        }

        final_flux_energy = 0;
        for (int i = 0; i < num_reactions; i++) {
            final_flux_energy += flux[i] * q_value[i];
        }

        t += dt;
        dt = 1e-4 * t;
        if (t + dt > *tstep) {
            dt = fmax(*tstep - t, 1e-30);
        }
    }

    double normalfac = 0;
    for (int i = 0; i < SIZE; i++) {
        normalfac += xout[i] * aa[i];
    }

    // Match the current GPU kernel's end-only renormalization, but return mass
    // fractions here because CPU callers and tests expect xout in X-space.
    double norm_sum = 1 / normalfac;
    __DIAG_HALT("Normalization", "normalfac", &norm_sum, 1);
    for (int i = 0; i < SIZE; i++) {
        xout[i] = xout[i] * aa[i] * norm_sum;
    }

    // Match the current GPU kernel's final-flux energy calculation.
    *sdotrate = final_flux_energy * (*tstep) * 9.5768e17;

    return;
}

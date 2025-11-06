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

// Interpretation from Fortran call:
//
//   hyperion_burner(tstep, temp, dens, xin(:), xout(:), ..., size)
//
//   `size` refers to the size of the xin & xout vectors
//
// I don't know what the other vars are...

void hyperion_burner_(double* tstep, double* temp, double* dens, double* xin,
                      double* restrict xout, double* sdotrate,
                      uchar* burned_zone, int* size) {

    memcpy(xout, xin, *size * sizeof(double));

    // We use the  XOUT memory, but our computation is with abundance (y)
    // Yes, this is confusing.
    for (int i = 0; i < *size; i++) {
        xout[i] = xout[i] / aa[i];
    }
    
    __DIAG_HALT("xin->xout", "xout", xout, *size);

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
            // integration. ESPECIALY since it's the biggest time-cost. If we do
            // that, it may be possible to build a very optimized routine for
            // it.
            int min_plus = 0;
            int min_minus = 0;
            if (i > 0) {
                min_plus = f_plus_max[i - 1] + 1;
                min_minus = f_minus_max[i - 1] + 1;
            }
            f_plus_sum[i] = 0.0;
            for (int j = min_plus; j <= f_plus_max[i]; j++) {
                f_plus_sum[i] += f_plus[j];
            }
            f_minus_sum[i] = 0.0;
            for (int j = min_minus; j <= f_minus_max[i]; j++) {
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

                // NOTE: FMA is slower here (I suspect due to the blocking of
                // the accumulator)
            } else {
                // Forward Euler update.
                xout[i] += (f_plus_sum[i] - f_minus_sum[i]) * dt;
            }
        }
        __DIAG_HALT("Y step", "xout", xout, *size);

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
        dt = 1e-2 * t; // 1% acc code
    }

    for (int i = 0; i < *size; i++) {
        xout[i] = xout[i] * aa[i];
    }

    *sdotrate = ddt_e;

    return;
}

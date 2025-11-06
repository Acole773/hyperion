#include "bn_burner_gpu.h"

#define ZERO_FIX 1e-50
#define THIRD 1.0 / 3.0

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

#define SIZE 16

#if SIZE == 16
#define SIZE 16
#define NUM_REACTIONS 48
#define NUM_FLUXES_PLUS 72
#define NUM_FLUXES_MINUS 72
#endif

#if SIZE == 150
#define SIZE 150
#define NUM_REACTIONS 1604
// TODO: find the actual value for these...
#define NUM_FLUXES_PLUS 2710
#define NUM_FLUXES_MINUS 2704
#endif

#if SIZE == 365
#define SIZE 365
#define NUM_REACTIONS 4395
#define NUM_FLUXES_PLUS 7429
#define NUM_FLUXES_MINUS 7420
#endif

extern "C" __global__ void hyperion_burner_dev_kernel(
    double* temp, double* dens, double* xin, double* xout, double* sdotrate,
    unsigned char* burned_zone, double* prefactor, double* p_0, double* p_1,
    double* p_2, double* p_3, double* p_4, double* p_5, double* p_6, double* aa,
    double* q_value, int* reactant_1, int* reactant_2, int* reactant_3,
    int* f_plus_map, int* f_minus_map, double* f_plus_factor,
    double* f_minus_factor, int* f_plus_max, int* f_minus_max,
    int* num_react_species, int* int_vals, double* real_vals) {

    // xout = xin;
    // No need to separate since we are only GPU. This does both transfer and
    // set.
    int n = threadIdx.x;
    int i = n; // for loop on threads
    // if (n > 0) {
    // return;
    // }

    // for (int i = 0; i < SIZE; i++) {
    if (i < SIZE) {
        xout[i] = xin[i] / aa[i];
    }

    double tmp = temp[0] / 1e9;
    double ddt_e = 0;

    double t93 = pow(tmp, THIRD);
    double t1 = 1 / tmp;
    double t2 = 1 / t93;
    double t3 = t93;
    double t4 = tmp;
    double t5 = t93 * t93 * t93 * t93 * t93;
    double t6 = log(tmp);

    double density[3] = {1.0, dens[0], (dens[0]) * (dens[0])};

    // double rate[NUM_REACTIONS];
    // double flux[NUM_REACTIONS];
    // double f_plus[NUM_FLUXES_PLUS];
    // double f_minus[NUM_FLUXES_MINUS];
    // double f_plus_sum[SIZE];
    // double f_minus_sum[SIZE];
    __shared__ double rate[NUM_REACTIONS];
    __shared__ double flux[NUM_REACTIONS];
    __shared__ double f_plus[NUM_FLUXES_PLUS];
    __shared__ double f_minus[NUM_FLUXES_MINUS];
    __shared__ double f_plus_sum[SIZE];
    __shared__ double f_minus_sum[SIZE];

    __syncthreads();

    // for (int i = 0; i < NUM_REACTIONS; i++) {
    if (i < NUM_REACTIONS) {
        double prefactor_ = prefactor[i] * density[num_react_species[i] - 1];
        rate[i] =
            prefactor_ * exp(p_0[i] + t1 * p_1[i] + t2 * p_2[i] + t3 * p_3[i] +
                             t4 * p_4[i] + t5 * p_5[i] + t6 * p_6[i]);
    }

    double t = 1e-20;
    double dt = t * 1e-2;

    double tstep = real_vals[KSR_TSTEP];

    while (t < tstep) {

        // for (int i = 0; i < NUM_REACTIONS; i++) {
        if (i < NUM_REACTIONS) {
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

        __syncthreads();

        // for (int i = 0; i < NUM_FLUXES_PLUS; i++) {
        if (i < NUM_FLUXES_PLUS) {
            f_plus[i] = f_plus_factor[i] * flux[f_plus_map[i]];
        }

        // for (int i = 0; i < NUM_FLUXES_MINUS; i++) {
        if (i < NUM_FLUXES_MINUS) {
            f_minus[i] = f_minus_factor[i] * flux[f_minus_map[i]];
        }

        __syncthreads();

        // for (int i = 0; i < SIZE; i++) {
        if (i < SIZE) {
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

        __syncthreads();

        // for (int i = 0; i < SIZE; i++) {
        if (i < SIZE) {
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

        // TODO: no renorm or ddt_e calc since they are unpleasant to parallelize. We will think about the best way to remedy that issue later.

        // double normalfac = 0;
        // // for (int i = 0; i < SIZE; i++) {
        // if (i < SIZE) {
        //     xout[i] = xout[i] * aa[i];
        //     normalfac += xout[i];
        // }
        //
        // normalfac = 1 / normalfac;
        // // for (int i = 0; i < SIZE; i++) {
        // if (i < SIZE) {
        //     xout[i] = xout[i] * normalfac;
        //     xout[i] = xout[i] / aa[i];
        // }

        // tmp = 0;
        // // for (int i = 0; i < NUM_REACTIONS; i++) {
        // if (i < NUM_REACTIONS) {
        //     tmp += flux[i] * q_value[i];
        // }
        // tmp *= dt;
        // ddt_e += tmp * 9.5768e17; // Convert MeV/nucleon/s to erg/g/s

        t += dt;
        dt = 1e-2 * t; // 1% acc code

        __syncthreads();
    }

    // for (int i = 0; i < SIZE; i++) {
    if (i < SIZE) {
        xout[i] = xout[i] * aa[i];
    }

    sdotrate[0] = ddt_e;

    return;
}

#include "bn_burner_gpu.h"

#define ZERO_FIX 1e-50
#define THIRD 1.0 / 3.0

#include <hip/hip_runtime.h>

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

extern "C" __global__ void hyperion_burner_dev_kernel(
    double* temp, double* dens, double* xin, double* xout, double* sdotrate,
    double* prefactor, double* p_0, double* p_1,
    double* p_2, double* p_3, double* p_4, double* p_5, double* p_6, double* aa,
    double* q_value, int* reactant_1, int* reactant_2, int* reactant_3,
    int* f_plus_map, int* f_minus_map, double* f_plus_factor,
    double* f_minus_factor, int* f_plus_max, int* f_minus_max,
    int* num_react_species, double* real_vals) {

    const int tid = threadIdx.x;

    // ------------------------------------------------------------
    // Shared memory
    // ------------------------------------------------------------
    __shared__ double rate[NUM_REACTIONS];
    __shared__ double flux[NUM_REACTIONS];
    __shared__ double f_plus[NUM_FLUXES_PLUS];
    __shared__ double f_minus[NUM_FLUXES_MINUS];
    __shared__ double f_plus_sum[SIZE];
    __shared__ double f_minus_sum[SIZE];
    __shared__ double norm_sum;
    __shared__ double energy_sum;

    __shared__ double reduce_buf[1024];  // must be >= blockDim.x



    // ------------------------------------------------------------
    // Initial normalization: xout = xin / aa
    // ------------------------------------------------------------
    if (tid < SIZE)
        xout[tid] = xin[tid] / aa[tid];

    // ------------------------------------------------------------
    // Thermodynamic precompute
    // ------------------------------------------------------------
    double tmp = temp[0] * 1e-9;
    double t93 = pow(tmp, THIRD);

    double t1 = 1.0 / tmp;
    double t2 = 1.0 / t93;
    double t3 = t93;
    double t4 = tmp;
    double t5 = pow(t93, 5);
    double t6 = log(tmp);

    double density[3] = {
        1.0,
        dens[0],
        dens[0] * dens[0]
    };

    __syncthreads();

    // ------------------------------------------------------------
    // Rate evaluation
    // ------------------------------------------------------------
    if (tid < NUM_REACTIONS) {
        int ns = num_react_species[tid] - 1;
        rate[tid] =
            prefactor[tid] * density[ns] *
            exp(p_0[tid]
              + p_1[tid] * t1
              + p_2[tid] * t2
              + p_3[tid] * t3
              + p_4[tid] * t4
              + p_5[tid] * t5
              + p_6[tid] * t6);
    }
    __syncthreads();

    // ------------------------------------------------------------
    // Time integration
    // ------------------------------------------------------------
    double t     = 1e-20;
    double dt    = t * 1e-2;
    double tstep = real_vals[KSR_TSTEP];

    while (t < tstep) {


        // --------------------------------------------------------
        // Flux computation
        // --------------------------------------------------------
        if (tid < NUM_REACTIONS) {
            double f = rate[tid];
            if (num_react_species[tid] >= 1)
                f *= xout[reactant_1[tid]];
            if (num_react_species[tid] >= 2)
                f *= xout[reactant_2[tid]];
            if (num_react_species[tid] == 3)
                f *= xout[reactant_3[tid]];
            flux[tid] = f;
        }

        __syncthreads();

        // --------------------------------------------------------
        // Expand fluxes
        // --------------------------------------------------------
        if (tid < NUM_FLUXES_PLUS)
            f_plus[tid] = f_plus_factor[tid] * flux[f_plus_map[tid]];

        if (tid < NUM_FLUXES_MINUS)
            f_minus[tid] = f_minus_factor[tid] * flux[f_minus_map[tid]];

        __syncthreads();

        // --------------------------------------------------------
        // Species updates
        // --------------------------------------------------------
        if (tid < SIZE) {
            int p0 = (tid == 0) ? 0 : f_plus_max[tid - 1] + 1;
            int m0 = (tid == 0) ? 0 : f_minus_max[tid - 1] + 1;

            double plus  = 0.0;
            double minus = 0.0;

            for (int j = p0; j <= f_plus_max[tid]; ++j)
                plus += f_plus[j];

            for (int j = m0; j <= f_minus_max[tid]; ++j)
                minus += f_minus[j];

            if (minus * dt > xout[tid]) {
                xout[tid] =
                    (xout[tid] + plus * dt) /
                    (1.0 + (minus / (xout[tid] + ZERO_FIX)) * dt);
            } else {
                xout[tid] += (plus - minus) * dt;
            }
        }

        __syncthreads();

        t += dt;
        dt = 1e-4 * t;

    }

    // ------------------------------------------------------------
    // Final renormalization (block reduction)
    // ------------------------------------------------------------
    double local = 0.0;
    if (tid < SIZE)
        local = xout[tid] * aa[tid];

    reduce_buf[tid] = local;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduce_buf[tid] += reduce_buf[tid + stride];
       __syncthreads();
    }

    if (tid == 0)
        norm_sum = 1.0 / reduce_buf[0];

    __syncthreads();

    if (tid < SIZE)
        xout[tid] = (xout[tid] * aa[tid] * norm_sum) / aa[tid];



    // ------------------------------------------------------------
    // Energy generation (block reduction)
    // ------------------------------------------------------------

    double e_local = 0.0;
    if (tid < NUM_REACTIONS)
        e_local = flux[tid] * q_value[tid];

    reduce_buf[tid] = e_local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduce_buf[tid] += reduce_buf[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        sdotrate[0] = reduce_buf[0] * dt * 9.5768e17;
}

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
    int* num_react_species, double* real_vals, double* rate_g, double* flux_g) {

    int zone = blockIdx.x;
    int tid = threadIdx.x;

    double* rate = rate_g + zone * NUM_REACTIONS;
    double* flux = flux_g + zone * NUM_REACTIONS;

    double tstep = real_vals[KSR_TSTEP];

    double T   = temp[zone];
    double rho = dens[zone];

    // Per-zone offsets
    double* xout_zone = xout + zone * SIZE;
    double* xin_zone  = xin  + zone * SIZE;
    double* aa_zone   = aa   + zone * SIZE;

    // ------------------------------------------------------------
    // Shared memory
    // ------------------------------------------------------------
    extern __shared__ double smem[];
    double* f_plus_sum  = smem;
    double* f_minus_sum = f_plus_sum  + SIZE;

    __shared__ double norm_sum;
    __shared__ double energy_sum;

    __shared__ double reduce_buf[256]; 

    __shared__ double t;
    __shared__ double dt;


    // ------------------------------------------------------------
    // Initial normalization: xout = xin / aa
    // ------------------------------------------------------------
    for (int i = tid; i < SIZE; i += blockDim.x)
        xout_zone[i] = xin_zone[i] / aa_zone[i];

    // ------------------------------------------------------------
    // Thermodynamic precompute
    // ------------------------------------------------------------
    double tmp = T * 1e-9;
    double t93 = pow(tmp, THIRD);

    double t1 = 1.0 / tmp;
    double t2 = 1.0 / t93;
    double t3 = t93;
    double t4 = tmp;
    double t93_2 = t93 * t93;
    double t5 = t93_2 * t93_2 * t93;
    double t6 = log(tmp);

    double density[3];
    density[0] = 1.0;
    density[1] = rho;
    density[2] = rho * rho;

    // ------------------------------------------------------------
    // Rate evaluation
    // ------------------------------------------------------------
    for (int r = tid; r < NUM_REACTIONS; r += blockDim.x) {
        int ns = num_react_species[r] - 1;
        ns = max(0, min(ns, 2));

        rate[r] =
            prefactor[r] * density[ns] *
            exp(p_0[r]
              + p_1[r] * t1
              + p_2[r] * t2
              + p_3[r] * t3
              + p_4[r] * t4
              + p_5[r] * t5
              + p_6[r] * t6);
    }
    __syncthreads();

    // ------------------------------------------------------------
    // Time integration
    // ------------------------------------------------------------

    if (tid == 0) {
        t  = 1e-20;
        dt = t * 1e-2;
    }
    __syncthreads(); // Ensure every thread sees the initial t and dt

    while (t < tstep) {


        // --------------------------------------------------------
        // Flux computation
        // --------------------------------------------------------
        for (int r = tid; r < NUM_REACTIONS; r += blockDim.x) {
            double f = rate[r];
	    if (num_react_species[r] > 0) f *= xout_zone[reactant_1[r]];
	    if (num_react_species[r] > 1) f *= xout_zone[reactant_2[r]];
	    if (num_react_species[r] > 2) f *= xout_zone[reactant_3[r]];
            flux[r] = f;
        }

        __syncthreads();

        // --------------------------------------------------------
        // Species updates
        // --------------------------------------------------------
        for (int i = tid; i < SIZE; i += blockDim.x) {
            int p0 = (i == 0) ? 0 : f_plus_max[i - 1] + 1;
            int m0 = (i == 0) ? 0 : f_minus_max[i - 1] + 1;
            double plus  = 0.0;
	    double minus = 0.0;

            for (int j = p0; j <= f_plus_max[i]; ++j)
	        plus += f_plus_factor[j] * flux[f_plus_map[j]];
	    for (int j = m0; j <= f_minus_max[i]; ++j)
		minus += f_minus_factor[j] * flux[f_minus_map[j]];


            if (minus * dt > xout_zone[i]) {
		// Asympotic estimate.
                xout_zone[i] =
                    (xout_zone[i] + plus * dt) /
                    (1.0 + (minus / (xout_zone[i] + ZERO_FIX)) * dt);
            } else {
		// Forward Euler update.
                xout_zone[i] += (plus - minus) * dt;
            }
        }
  	if (tid == 0) {
            t += dt;              // ADVANCE time, do not reset it!
        
        dt = 1e-4 * t;        
        
        // 4. Boundary Condition (Safety check to not overshoot tstep)
        if (t + dt > tstep) {
            dt = tstep - t;
        }
	}
	__syncthreads();
    }

    // ------------------------------------------------------------
    // Final renormalization (block reduction)
    // ------------------------------------------------------------
    double local = 0.0;
    for (int i = tid; i < SIZE; i += blockDim.x)
        local += xout_zone[i] * aa_zone[i];

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

    for (int i = tid; i < SIZE; i += blockDim.x)
        xout_zone[i] = (xout_zone[i] * aa_zone[i] * norm_sum) / aa_zone[i];



    // ------------------------------------------------------------
    // Energy generation (block reduction)
    // ------------------------------------------------------------

    double e_local = 0.0;
    for (int r = tid; r < NUM_REACTIONS; r += blockDim.x)
        e_local += flux[r] * q_value[r];

    reduce_buf[tid] = e_local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduce_buf[tid] += reduce_buf[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        sdotrate[0] = reduce_buf[0] * tstep * 9.5768e17;
}

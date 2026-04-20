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

#include <math.h>

#define GPU_ASSERT_FINITE(val, tag, species_idx, flux_idx)             \
    do {                                                               \
        if (!isfinite(val)) {                                          \
	    printf("CRITICAL ERROR: %s is NaN at Species: %d, Flux_Map_Idx: %d\n", \
               tag, species_idx, flux_idx); \
            printf("  zone=%d thread=%d block=%d\n",                   \
                   blockIdx.x, threadIdx.x, blockIdx.x);               \
            printf("  value=%e\n", (double)(val));                     \
            /* __builtin_trap(); <-- Comment this out to prevent hardware exception */ \
            /* We print only the first failure to avoid "ridiculous prints" */ \
	   return; \
        }                                                              \
    } while (0)

#define GPU_DEBUG 1

#if GPU_DEBUG
#define GPU_GUARD(cond, msg, a, b, c) \
    if (cond) { \
        printf("GPU ERROR: %s | a=%d b=%d c=%d\n", msg, a, b, c); \
        asm("s_trap 0"); \
    }
#else
#define GPU_GUARD(cond, msg, a, b, c)
#endif

extern "C" __global__ void hyperion_burner_dev_kernel(
    int Nzones,
    double* temp, double* dens, double* xin, double* xout, double* sdotrate,
    double* prefactor, double* p_0, double* p_1,
    double* p_2, double* p_3, double* p_4, double* p_5, double* p_6, double* aa,
    double* q_value, int* reactant_1, int* reactant_2, int* reactant_3,
    int* f_plus_map, int* f_minus_map, double* f_plus_factor,
    double* f_minus_factor, int* f_plus_max, int* f_minus_max,
    int* num_react_species, double* real_vals, double* rate_g){

    int tid = threadIdx.x;

   
    for (int zone = blockIdx.x; zone < Nzones; zone += gridDim.x){ 

        double* rate = rate_g + zone * NUM_REACTIONS;
    
        double T   = temp[zone];
        double rho = dens[zone];
        double tstep = real_vals[0];
    
        double* aa_zone = aa;
    
        // Per-zone offsets
        double* xout_zone = xout + zone * SIZE;
        double* xin_zone  = xin  + zone * SIZE;
    
        // ------------------------------------------------------------
        // Shared memory
        // ------------------------------------------------------------
        extern __shared__ double smem[];
    
        double* flux       = smem;                        // NUM_REACTIONS
        double* reduce_buf = flux + NUM_REACTIONS;	      // blockDim.x
    
        __shared__ double norm_sum;
        double t, dt, t_end;
        int done;
    
        // ------------------------------------------------------------
        // Initial normalization: xout = xin / aa
        // ------------------------------------------------------------
        for (int i = tid; i < SIZE; i += blockDim.x)
            xout_zone[i] = xin_zone[i] / aa_zone[i];
    
        __syncthreads();
        // ------------------------------------------------------------
        // Thermodynamic precompute
        // ------------------------------------------------------------
        double tmp = T * 1e-9;
        double t93 = pow(tmp, THIRD);
    
        double t1 = 1.0 / (tmp);
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
    	//GPU_ASSERT_FINITE(rate[r], "rate");
        }
        __syncthreads();
    
        // ------------------------------------------------------------
        // Time integration
        // ------------------------------------------------------------
    
        
        t  = 1e-20;
        dt = t * 1e-2;
        t_end = tstep;
        done = 0;
    
        while (!done) {
    
            // --------------------------------------------------------
            // Flux computation
            // --------------------------------------------------------
            for (int r = tid; r < NUM_REACTIONS; r += blockDim.x) {
                double f_rate = rate[r];
    	    // Check if the rate itself is the problem (e.g., T-dependent math failure)
    	    if (!isfinite(f_rate)) {
    	        printf("ZONE %d: Reaction %d RATE is NaN\n", blockIdx.x, r);
    	    }
    	    double f = f_rate;
    	    if (num_react_species[r] > 0) {
    	        int idx1 = reactant_1[r];
    	        double abund1 = xout_zone[idx1];
    	        f *= abund1;
    	    }
    	   // if (num_react_species[r] > 0) f *= xout_zone[reactant_1[r]];
                if (num_react_species[r] > 1) {
                    int idx2 = reactant_2[r];
                    double abund2 = xout_zone[idx2];
                    f *= abund2;
                }
    
    	   // if (num_react_species[r] > 1) f *= xout_zone[reactant_2[r]];
                if (num_react_species[r] > 2) {
                    int idx3 = reactant_3[r];
                    double abund3 = xout_zone[idx3];
                    f *= abund3;
                }
    
    	   // if (num_react_species[r] > 2) f *= xout_zone[reactant_3[r]];
                flux[r] = f;
            }
    
            __syncthreads(); //this ensures all fluxes have been updated
    
            // --------------------------------------------------------
            // Species updates
            // --------------------------------------------------------
            for (int i = tid; i < SIZE; i += blockDim.x) {
                // f_*_max has a sentinel at [0] (= -1) and the old "upper
                // bound for species i" now lives at [i + 1]; this removes the
                // (i == 0) conditional that was previously needed.
                int p0 = f_plus_max[i] + 1;
                int m0 = f_minus_max[i] + 1;
                double plus  = 0.0;
	        double minus = 0.0;

                for (int j = p0; j <= f_plus_max[i + 1]; ++j)         //this could be source of divergence
	            plus += f_plus_factor[j] * flux[f_plus_map[j]];
	        for (int j = m0; j <= f_minus_max[i + 1]; ++j)        // ''
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
                // NEW: clamp
                if (xout_zone[i] < ZERO_FIX)
                    xout_zone[i] = ZERO_FIX;
            }
            __syncthreads();
    
            t += dt;        
            dt = 1e-4 * t;
    	if (t + dt > t_end)
    	     dt = max(t_end - t, 1e-30);        
    
            if (t >= t_end || !isfinite(t) || dt <= 0.0)
                done = 1;
        }

    
    
        // ------------------------------------------------------------
        // Final renormalization (block reduction)
        //   Phase 1: intra-wave reduction via warp shuffle
        //   Phase 2: cross-wave reduction via LDS
        // ------------------------------------------------------------
        double local = 0.0;
        for (int i = tid; i < SIZE; i += blockDim.x)
            local += xout_zone[i] * aa_zone[i];

        // Intra-wave reduction (wavefront = 64 lanes on gfx90a)
        for (int offset = 32; offset > 0; offset >>= 1)
            local += __shfl_down(local, offset, 64);

        int lane   = tid & 63;
        int waveId = tid >> 6;

        if (lane == 0)
            reduce_buf[waveId] = local;

        __syncthreads();

        // Cross-wave reduction: wave 0 reduces the 4 partial sums
        if (tid < 4) {
            local = reduce_buf[tid];
            for (int offset = 2; offset > 0; offset >>= 1)
                local += __shfl_down(local, offset, 64);
        }

        if (tid == 0)
            norm_sum = 1.0 / local;

        __syncthreads();

        for (int i = tid; i < SIZE; i += blockDim.x)
            xout_zone[i] = (xout_zone[i] * aa_zone[i] * norm_sum) / aa_zone[i];

        __syncthreads();


        // ------------------------------------------------------------
        // Energy generation (block reduction)
        //   Phase 1: intra-wave reduction via warp shuffle
        //   Phase 2: cross-wave reduction via LDS
        // ------------------------------------------------------------

        double e_local = 0.0;
        for (int r = tid; r < NUM_REACTIONS; r += blockDim.x)
            e_local += flux[r] * q_value[r];

        for (int offset = 32; offset > 0; offset >>= 1)
            e_local += __shfl_down(e_local, offset, 64);

        if (lane == 0)
            reduce_buf[waveId] = e_local;

        __syncthreads();

        if (tid < 4) {
            e_local = reduce_buf[tid];
            for (int offset = 2; offset > 0; offset >>= 1)
                e_local += __shfl_down(e_local, offset, 64);
        }

        if (tid == 0)
            sdotrate[zone] = e_local * tstep * 9.5768e17;

         __syncthreads();
    }
}

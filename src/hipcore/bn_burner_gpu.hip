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

   
    // LDS layout (dynamic shared memory):
    //   flux       [NUM_REACTIONS]  - exp06 baseline
    //   reduce_buf [num_waves]      - block-level reduction scratch
    //   rate       [NUM_REACTIONS]  - exp06: moved from global rate_g
    //   xout_lds   [SIZE]           - exp09: moved from global xout_zone.
    //   nrs_lds    [NUM_REACTIONS]  - exp10: packed num_react_species as
    //                                 unsigned char (values 1..3). Saves 4x
    //                                 vs int. Read once per reaction in flux
    //                                 loop, no need for the global reload.
    const int num_waves_k = blockDim.x >> 6;
    extern __shared__ double smem[];
    double* flux       = smem;                              // NUM_REACTIONS
    double* reduce_buf = flux + NUM_REACTIONS;              // num_waves
    double* rate       = reduce_buf + num_waves_k;          // NUM_REACTIONS
    double* xout_lds   = rate + NUM_REACTIONS;              // SIZE
    unsigned char* nrs_lds  = (unsigned char*)(xout_lds + SIZE);  // NUM_REACTIONS bytes
    unsigned char* r1_lds   = nrs_lds + NUM_REACTIONS;            // NUM_REACTIONS bytes
    unsigned char* r2_lds   = r1_lds  + NUM_REACTIONS;            // NUM_REACTIONS bytes
    unsigned char* r3_lds   = r2_lds  + NUM_REACTIONS;            // NUM_REACTIONS bytes
    unsigned short* fp_max_lds = (unsigned short*)(r3_lds + NUM_REACTIONS); // SIZE
    unsigned short* fm_max_lds = fp_max_lds + SIZE;                         // SIZE
    unsigned short* fp_map_lds = fm_max_lds + SIZE;                         // NUM_FLUXES_PLUS
    unsigned short* fm_map_lds = fp_map_lds + NUM_FLUXES_PLUS;              // NUM_FLUXES_MINUS
    // Pad to 8-byte alignment for the double aa_lds that follows.
    double* aa_lds = (double*)(((uintptr_t)(fm_map_lds + NUM_FLUXES_MINUS) + 7) & ~(uintptr_t)7);  // SIZE doubles
    // exp11/12/13: pack reactant_1/2/3 as uchar (species idx < SIZE = 150).
    // exp14: pack f_plus_max / f_minus_max as ushort.
    // exp15: pack f_plus_map / f_minus_map as ushort. Values are reaction
    //        indices in [0, NUM_REACTIONS=1604) so ushort suffices.

    // Stage block-shared metadata into LDS once per block.
    for (int r = tid; r < NUM_REACTIONS; r += blockDim.x) {
        nrs_lds[r] = (unsigned char)num_react_species[r];
        r1_lds[r]  = (unsigned char)reactant_1[r];
        r2_lds[r]  = (unsigned char)reactant_2[r];
        r3_lds[r]  = (unsigned char)reactant_3[r];
    }
    for (int i = tid; i < SIZE; i += blockDim.x) {
        fp_max_lds[i] = (unsigned short)f_plus_max[i];
        fm_max_lds[i] = (unsigned short)f_minus_max[i];
    }
    for (int j = tid; j < NUM_FLUXES_PLUS; j += blockDim.x)
        fp_map_lds[j] = (unsigned short)f_plus_map[j];
    for (int j = tid; j < NUM_FLUXES_MINUS; j += blockDim.x)
        fm_map_lds[j] = (unsigned short)f_minus_map[j];
    // exp16: aa[SIZE] is read in initial-norm, energy-reduce and final-norm
    //        - dozens of times per zone per timestep. Tiny LDS cost (1.2KB).
    for (int i = tid; i < SIZE; i += blockDim.x)
        aa_lds[i] = aa[i];
    __syncthreads();

    for (int zone = blockIdx.x; zone < Nzones; zone += gridDim.x){ 

        double T   = temp[zone];
        double rho = dens[zone];
        double tstep = real_vals[0];
    
        double* aa_zone = aa_lds;  // exp16: redirect to LDS copy
    
        // Per-zone offsets
        // Experiment 09: xout_zone now lives in LDS for the duration of the
        // per-zone work (writeback to global at end of the zone loop body).
        double* xout_zone = xout_lds;
        double* xin_zone  = xin  + zone * SIZE;
    
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
        // Experiment 01: pow(tmp, 1/3) -> cbrt(tmp)
        // pow() is inlined as exp(y*log(x)) with NaN/sign/integer-y special cases
        // (~335 lines of gfx90a asm at baseline). cbrt() is a dedicated intrinsic
        // with a much shorter sequence, reducing VGPR/SGPR pressure.
        double t93 = cbrt(tmp);
    
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
    	    int nrs = (int)nrs_lds[r];
    	    if (nrs > 0) {
    	        int idx1 = (int)r1_lds[r];
    	        double abund1 = xout_zone[idx1];
    	        f *= abund1;
    	    }
                if (nrs > 1) {
                    int idx2 = (int)r2_lds[r];
                    double abund2 = xout_zone[idx2];
                    f *= abund2;
                }
                if (nrs > 2) {
                    int idx3 = (int)r3_lds[r];
                    double abund3 = xout_zone[idx3];
                    f *= abund3;
                }
                flux[r] = f;
            }
    
            __syncthreads(); //this ensures all fluxes have been updated
    
            // --------------------------------------------------------
            // Species updates
            // --------------------------------------------------------
            for (int i = tid; i < SIZE; i += blockDim.x) {
                int p0 = (i == 0) ? 0 : (int)fp_max_lds[i - 1] + 1;
                int m0 = (i == 0) ? 0 : (int)fm_max_lds[i - 1] + 1;
                int p1 = (int)fp_max_lds[i];   // exp14: hoist loop-invariant bound
                int m1 = (int)fm_max_lds[i];
                double plus  = 0.0;
    	        double minus = 0.0;
    
                for (int j = p0; j <= p1; ++j)
    	            plus += f_plus_factor[j] * flux[(int)fp_map_lds[j]];
    	        for (int j = m0; j <= m1; ++j)
    		    minus += f_minus_factor[j] * flux[(int)fm_map_lds[j]];
    
    
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
        const int num_waves = blockDim.x >> 6;

        if (lane == 0)
            reduce_buf[waveId] = local;

        __syncthreads();

        // Cross-wave reduction: wave 0 reduces the num_waves partial sums
        if (tid < num_waves) {
            local = reduce_buf[tid];
            for (int offset = num_waves >> 1; offset > 0; offset >>= 1)
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

        if (tid < num_waves) {
            e_local = reduce_buf[tid];
            for (int offset = num_waves >> 1; offset > 0; offset >>= 1)
                e_local += __shfl_down(e_local, offset, 64);
        }

        if (tid == 0)
            sdotrate[zone] = e_local * tstep * 9.5768e17;

         __syncthreads();

        // Experiment 09: copy LDS xout_lds back to global xout for the host.
        double* xout_global = xout + zone * SIZE;
        for (int i = tid; i < SIZE; i += blockDim.x)
            xout_global[i] = xout_lds[i];

        __syncthreads();
    }
}

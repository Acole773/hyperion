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

#define USE_SHUFFLE_BLOCK_REDUCTION 1

#include <math.h>

#define GPU_ASSERT_FINITE(val, tag, species_idx, flux_idx)             \
    do {                                                               \
        if (!isfinite(val)) {                                          \
            printf("CRITICAL ERROR: %s is NaN at Species: %d, Flux_Map_Idx: %d\n", \
               tag, species_idx, flux_idx); \
            printf("  zone=%d thread=%d block=%d\n",                   \
                   blockIdx.x, threadIdx.x, blockIdx.x);               \
            printf("  value=%e\n", (double)(val));                     \
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

#define RF(r_,j_) reactant_filter[3*(r_) + (j_)]

#ifndef USE_SHUFFLE_BLOCK_REDUCTION
#define USE_SHUFFLE_BLOCK_REDUCTION 1
#endif


#define SMALL_POW(x, n) ({ \
    double _x = (double)(x); \
    int _n = (int)(n); \
    (_n == 2) ? (_x * _x) : \
    (_n == 1) ? (_x) : \
    (1.0); \
})

__device__ inline double block_reduce_sum_tree(double val, double* smem)
{
    const int tid = threadIdx.x;

    smem[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    return smem[0];
}

__device__ inline double wave_reduce_sum(double val)
{
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, warpSize);
    }
    return val;
}

__device__ inline double block_reduce_sum_shuffle(double val, double* smem)
{
    const int tid   = threadIdx.x;
    const int lane  = tid % warpSize;
    const int wid   = tid / warpSize;
    const int nwave = (blockDim.x + warpSize - 1) / warpSize;

    val = wave_reduce_sum(val);

    if (lane == 0) {
        smem[wid] = val;
    }
    __syncthreads();

    double block_sum = 0.0;
    if (wid == 0) {
        block_sum = (lane < nwave) ? smem[lane] : 0.0;
        block_sum = wave_reduce_sum(block_sum);
    }

    return block_sum;
}

__device__ inline double block_reduce_sum(double val, double* smem)
{
#if USE_SHUFFLE_BLOCK_REDUCTION
    return block_reduce_sum_shuffle(val, smem);
#else
    return block_reduce_sum_tree(val, smem);
#endif
}

extern "C" __global__ void hyperion_burner_dev_kernel(
    double* temp, double* dens, double* xin, double* xout, double* sdotrate,
    double* prefactor, double* p_0, double* p_1,
    double* p_2, double* p_3, double* p_4, double* p_5, double* p_6, double* aa,
    double* q_value, int* reactant_1, int* reactant_2, int* reactant_3, double* reactant_filter,
    int* f_plus_map, int* f_minus_map, double* f_plus_factor,
    double* f_minus_factor, int* f_plus_max, int* f_minus_max,
    int* num_react_species, double* real_vals, double* rate_g){

    int zone = blockIdx.x;
    int tid = threadIdx.x;

    double* rate = rate_g + zone * NUM_REACTIONS;

    //double T   = temp[zone];
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
    const double tmp = temp[zone] * 1e-9;
    const double t93 = pow(tmp, THIRD);

    // ------------------------------------------------------------
    // Rate evaluation
    // ------------------------------------------------------------
    for (int r = tid; r < NUM_REACTIONS; r += blockDim.x) {
        int ns = num_react_species[r] - 1;
        if (ns < 0) ns = 0;
        else if (ns > 2) ns = 2;

        rate[r] =
            prefactor[r] * SMALL_POW(rho,ns) *
            exp(p_0[r]
              + p_1[r] / tmp
              + p_2[r] / t93
              + p_3[r] * t93
              + p_4[r] * tmp
              + p_5[r] * t93 * t93 * t93 * t93 *t93
              + p_6[r] * log(tmp) );
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
            int idx1 = reactant_1[r];
            int idx2 = reactant_2[r];
            int idx3 = reactant_3[r];

            double a1 = xout_zone[idx1];
            double a2 = xout_zone[idx2];
            double a3 = xout_zone[idx3];

            double f1 = RF(r,0);
            double f2 = RF(r,1);
            double f3 = RF(r,2);

            double term1 = f1 * a1 + (1.0 - f1);
            double term2 = f2 * a2 + (1.0 - f2);
            double term3 = f3 * a3 + (1.0 - f3);

            flux[r] = rate[r] * term1 * term2 * term3;
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

            double x = xout_zone[i];
            double asym =
              (x + plus * dt) /
              (1.0 + (minus / (x + ZERO_FIX)) * dt);
            double euler = x + (plus - minus) * dt;
            x = (minus * dt > x) ? asym : euler;
            x = (x < ZERO_FIX) ? ZERO_FIX : x;
            xout_zone[i] = x;

            //GPU_ASSERT_FINITE(xout_zone[i], "xout", i, 0);
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
    // ------------------------------------------------------------
    double local = 0.0;
    for (int i = tid; i < SIZE; i += blockDim.x)
        local += xout_zone[i] * aa_zone[i];

    double sum = block_reduce_sum(local, reduce_buf);

    if (tid == 0)
        norm_sum = 1.0 / sum;

    __syncthreads();

    for (int i = tid; i < SIZE; i += blockDim.x)
        xout_zone[i] *= norm_sum;

    __syncthreads();

    // ------------------------------------------------------------
    // Energy generation (block reduction)
    // ------------------------------------------------------------
    double e_local = 0.0;
    for (int r = tid; r < NUM_REACTIONS; r += blockDim.x)
        e_local += flux[r] * q_value[r];

    double e_sum = block_reduce_sum(e_local, reduce_buf);

    if (tid == 0)
        sdotrate[zone] = e_sum * tstep * 9.5768e17;
}

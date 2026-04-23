#include "bn_burner_gpu.h"

#include "../core/store.h"
#include "../core/restrict.h"

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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
// #define SIZE 365
// #define NUM_REACTIONS 4395
// #define NUM_FLUXES_PLUS 7429
// #define NUM_FLUXES_MINUS 7420
#endif

// Global args structure for HIP device memory
static burner_args_t args;

// Forward declaration of the kernel wrapper
static void hyperion_burner_kernel(double* tstep, double* temp, double* dens,
                                   double* xin, double* xout, double* sdotrate,
                                   int zones);
double* d_rate = nullptr;
double* d_flux = nullptr;

// -----------------------------------------------------------------------------
// Entry point for Fortran-callable API
// -----------------------------------------------------------------------------
void hyperion_burner_(double* tstep, double* temp, double* dens, double* xin,
                      double* HYP_RESTRICT xout, double* sdotrate,
                      uchar* burned_zone, int* zones) {

        hyperion_burner_kernel(tstep, temp, dens, xin, xout, sdotrate, *zones);
}

// -----------------------------------------------------------------------------
// Allocate and initialize device memory for a batch of zones
// -----------------------------------------------------------------------------
int device_init(int zones) {
    hipError_t e;

    printf("[bn_burner_gpu] device_init called with zones=%d\n", zones);
    int error = 0;

    // exp35-prep: dump per-species j-count distribution (plus, minus, max).
    {
        int hist[16] = {0};
        int max_p = 0, max_m = 0, max_pm = 0;
        long sum_p = 0, sum_m = 0;
        for (int i = 0; i < num_species; i++) {
            int p = f_plus_max[i+1] - f_plus_max[i] - 1;
            int m = f_minus_max[i+1] - f_minus_max[i] - 1;
            int pm = (p > m) ? p : m;
            sum_p += p; sum_m += m;
            if (p > max_p) max_p = p;
            if (m > max_m) max_m = m;
            if (pm > max_pm) max_pm = pm;
            int b = pm / 8; if (b > 15) b = 15;
            hist[b]++;
        }
        printf("[jcount] num_species=%d sum_p=%ld sum_m=%ld avg_p=%.1f avg_m=%.1f max_p=%d max_m=%d max_pm=%d\n",
               num_species, sum_p, sum_m, (double)sum_p/num_species, (double)sum_m/num_species, max_p, max_m, max_pm);
        printf("[jcount] hist (8-wide bins of max(p,m)):");
        for (int b = 0; b < 16; b++) printf(" [%d-%d):%d", b*8, (b+1)*8, hist[b]);
        printf("\n");
    }

    // Allocate per-network arrays once (shared across zones)
    #define HIP_ALLOC_COPY(dest, src, n) \
        do { \
            error += hipMalloc(&(dest), (n) * sizeof(*(src))); \
            error += hipMemcpy(dest, src, (n) * sizeof(*(src)), hipMemcpyHostToDevice); \
        } while(0)

    HIP_ALLOC_COPY(args.prefactor, prefactor, num_reactions);
    HIP_ALLOC_COPY(args.p_0, p_0, num_reactions);
    HIP_ALLOC_COPY(args.p_1, p_1, num_reactions);
    HIP_ALLOC_COPY(args.p_2, p_2, num_reactions);
    HIP_ALLOC_COPY(args.p_3, p_3, num_reactions);
    HIP_ALLOC_COPY(args.p_4, p_4, num_reactions);
    HIP_ALLOC_COPY(args.p_5, p_5, num_reactions);
    HIP_ALLOC_COPY(args.p_6, p_6, num_reactions);
    HIP_ALLOC_COPY(args.aa, aa, num_species);
    HIP_ALLOC_COPY(args.q_value, q_value, num_reactions);
    HIP_ALLOC_COPY(args.reactant_1, reactant_1, num_reactions);
    HIP_ALLOC_COPY(args.reactant_2, reactant_2, num_reactions);
    HIP_ALLOC_COPY(args.reactant_3, reactant_3, num_reactions);
    HIP_ALLOC_COPY(args.f_plus_map, f_plus_map, f_plus_total);
    HIP_ALLOC_COPY(args.f_minus_map, f_minus_map, f_minus_total);
    HIP_ALLOC_COPY(args.f_plus_factor, f_plus_factor, f_plus_total);
    HIP_ALLOC_COPY(args.f_minus_factor, f_minus_factor, f_minus_total);
    HIP_ALLOC_COPY(args.f_plus_max, f_plus_max, num_species + 1);
    HIP_ALLOC_COPY(args.f_minus_max, f_minus_max, num_species + 1);
    HIP_ALLOC_COPY(args.num_react_species, num_react_species, num_reactions);

    // exp35: build a host permutation of species indices, sorted DESCENDING
    // by max(plus j-count, minus j-count). The kernel uses this to schedule
    // the "big" species (highest max-j) first via a wave-cooperative
    // (full 64-lane) reduction in Phase 1, and the "small" species via the
    // 16-lane sub-wave packing in Phase 2. The histogram for the 150-species
    // network is heavily skewed: 135 species have j<=15 (subwave-friendly),
    // 12 have j in [16,32), and 3 have j>=120 (max=403). In E32/E34 those
    // 3 huge species sit in random sub-wave groups and force the whole
    // wave to spin ceil(403/16)=26 inner iters with 3/4 of the lanes idle.
    // Pulling them into a wave-cooperative phase converts those wasted
    // lanes into useful reduction work (ceil(403/64)=7 inner iters with
    // ALL 64 lanes active).
    {
        int* perm = (int*)malloc(num_species * sizeof(int));
        int* jmax = (int*)malloc(num_species * sizeof(int));
        for (int i = 0; i < num_species; i++) {
            perm[i] = i;
            int p = f_plus_max[i+1] - f_plus_max[i] - 1;
            int m = f_minus_max[i+1] - f_minus_max[i] - 1;
            jmax[i] = (p > m) ? p : m;
        }
        // simple insertion sort (num_species ~150-365, called once)
        for (int i = 1; i < num_species; i++) {
            int kp = perm[i];
            int kj = jmax[kp];
            int j = i - 1;
            while (j >= 0 && jmax[perm[j]] < kj) {
                perm[j+1] = perm[j];
                j--;
            }
            perm[j+1] = kp;
        }
        printf("[exp35] species_perm sorted desc by max(p,m); top10:");
        for (int i = 0; i < 10 && i < num_species; i++)
            printf(" %d(%d)", perm[i], jmax[perm[i]]);
        printf(" ...bottom5:");
        for (int i = num_species - 5; i < num_species; i++)
            printf(" %d(%d)", perm[i], jmax[perm[i]]);
        printf("\n");

        e = hipMalloc(&args.species_perm, num_species * sizeof(int));
        if (e != hipSuccess) return EXIT_FAILURE;
        e = hipMemcpy(args.species_perm, perm, num_species * sizeof(int),
                      hipMemcpyHostToDevice);
        if (e != hipSuccess) return EXIT_FAILURE;

        free(perm);
        free(jmax);
    }

    // Per-zone allocations
    e = hipMalloc(&args.burned_zone, zones * sizeof(uchar)); if(e != hipSuccess) return EXIT_FAILURE;
    e = hipMalloc(&args.temp, zones * sizeof(double)); if(e != hipSuccess) return EXIT_FAILURE;
    e = hipMalloc(&args.dens, zones * sizeof(double)); if(e != hipSuccess) return EXIT_FAILURE;
    e = hipMalloc(&args.xin, zones * num_species * sizeof(double)); if(e != hipSuccess) return EXIT_FAILURE;
    e = hipMalloc(&args.xout, zones * num_species * sizeof(double)); if(e != hipSuccess) return EXIT_FAILURE;
    e = hipMalloc(&args.sdotrate, zones * sizeof(double)); if(e != hipSuccess) return EXIT_FAILURE;
    e = hipMalloc(&args.int_vals, zones * sizeof(int)); if(e != hipSuccess) return EXIT_FAILURE;
    e = hipMalloc(&args.real_vals, zones * sizeof(double)); if(e != hipSuccess) return EXIT_FAILURE;

    // Rate buffer
    e = hipMalloc(&d_rate, zones * NUM_REACTIONS * sizeof(double)); if(e != hipSuccess) return EXIT_FAILURE;
    e = hipMemset(d_rate, 0, zones * NUM_REACTIONS * sizeof(double)); if(e != hipSuccess) return EXIT_FAILURE;

    #undef HIP_ALLOC_COPY

    if (error > 0) {
        fprintf(stderr, "[bn_burner_gpu] device_init: HIP malloc/copy failed!\n");
        return EXIT_FAILURE;
    }

    printf("[bn_burner_gpu] device_init completed successfully.\n");
    return EXIT_SUCCESS;
}

// -----------------------------------------------------------------------------
// Kernel wrapper: copies per-zone data to device, launches kernel, copies back
// -----------------------------------------------------------------------------
static void hyperion_burner_kernel(double* tstep, double* temp, double* dens,
                                   double* xin, double* xout, double* sdotrate,
                                   int zones) {
    // Debug prints
    //printf("[bn_burner_gpu] hyperion_burner_kernel: zones=%d\n", zones);

    // Copy per-zone arrays to device
    hipMemcpy(args.temp, temp, zones * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(args.dens, dens, zones * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(args.xin, xin, zones * num_species * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(args.xout, xout, zones * num_species * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(args.sdotrate, sdotrate, zones * sizeof(double), hipMemcpyHostToDevice);
    // real_vals only contains scalar controls (not per-zone)
    hipMemcpy(
        args.real_vals,
        tstep,                 // host pointer
        sizeof(double),        // only one value
        hipMemcpyHostToDevice
    );

    // Kernel launch parameters
    // exp30: 256 -> 1024 (4 waves -> 16 waves per block). The E28
    // wave-cooperative species update has per-wave critical path
    // ~ (sum_of_j_counts_for_my_species / 64), so quadrupling the
    // wave count quarters the per-wave species count (~38 -> ~10) and
    // gives the SIMD scheduler 4x more in-flight waves to hide LDS /
    // HBM latency between issue slots. LDS budget per block is
    // unchanged (LDS is block-shared, not per-wave). VGPR cost
    // (16 * 116 = 1856 VGPRs / CU) still fits in the gfx90a 2048
    // VGPR/CU budget. Measured: wave occupancy 9% -> 49.7%, IPC
    // 0.35 -> 0.74, end-to-end -51.5% vs E28 (4 waves), -86.2%
    // cumulative vs original baseline. 1024 is the gfx90a hardware
    // max, cannot go higher.
    int bdim = 1024;
    {
        const char* env = getenv("HYP_BLOCKDIM");
        if (env) {
            bdim = atoi(env);
            if (bdim < 64) bdim = 64;
            if (bdim > 1024) bdim = 1024;
            bdim = (bdim / 64) * 64;  // round down to multiple of 64
        }
    }
    dim3 blockdim(bdim, 1, 1);
    int blocks = zones;
    dim3 griddim(blocks, 1, 1);
    int num_waves = blockdim.x / 64;

    // exp36/37: number of species processed in Phase 1 (wave-cooperative
    // 64-lane reduction). The 150-species network has exactly 3 "huge"
    // species (j = 403, 388, 328; ~30× the mean) and a cliff below
    // that — the next-largest has j=25. Anything that leaves even one
    // huge species in Phase 2 explodes: the worst sub-wave group in
    // Phase 2 blocks on ceil(403/16)=26 inner iters while three lanes
    // idle.
    //
    // Under the sequential-phase layout of E35/E36 the sweet spot was
    // phase1=4. With the CONCURRENT-phase layout introduced in E37
    // (waves <phase1_count run Phase 1, the rest run Phase 2 at the
    // same time) the block critical path becomes max(P1, P2) instead
    // of P1+P2, and wasting one wave on a medium species in Phase 1
    // now costs one full wave's worth of Phase 2 capacity. Re-sweep
    // (104 zones, 2 iter, concurrent E37):
    //
    //   phase1=3 -> 1.740 s   <-- optimum
    //   phase1=4 -> 1.885 s   (+8.3 %)
    //   phase1=5 -> 1.875 s   (+7.8 %)
    //   phase1=6 -> 1.901 s   (+9.3 %)
    //   phase1=8 -> 2.138 s   (+22.9 %)
    //
    // 3 is exactly the count of huge species — every wave beyond the
    // three that are strictly required for wave-coop is better spent
    // pushing Phase 2 along. Can be overridden by HYP_PHASE1 for
    // future re-tuning on different networks.
    int phase1_count = 3;
    {
        const char* env = getenv("HYP_PHASE1");
        if (env) phase1_count = atoi(env);
    }
    if (phase1_count > num_species) phase1_count = num_species;
    if (phase1_count < 0) phase1_count = 0;
    // Experiment 06: +NUM_REACTIONS doubles for LDS-resident `rate[]`.
    // Experiment 09: +SIZE doubles for LDS-resident `xout_zone[]`.
    // Experiment 10: +NUM_REACTIONS bytes for LDS-resident num_react_species
    //                packed as unsigned char (values are 0..3).
    // Experiment 11/12/13: +NUM_REACTIONS uchar each for reactant_1/2/3 LDS.
    // Experiment 14: +2*(SIZE+1) short for f_plus_max / f_minus_max LDS
    //                (SIZE+1 accounts for the leading sentinel at [0]).
    // Experiment 15: +NUM_FLUXES_{PLUS,MINUS} ushort for f_plus_map/f_minus_map LDS.
    // Experiment 16: +SIZE doubles for aa LDS (+ 8 bytes alignment slack).
    // Experiment 19: +(NUM_FLUXES_PLUS + NUM_FLUXES_MINUS) uchar for
    //                f_plus_factor / f_minus_factor LDS.
    // Experiment 25: xout_lds is now SIZE+1 doubles (+1 dummy slot for
    //                branchless unused-reactant masking).
    size_t sharedmem_allocation =
	sizeof(double) * (NUM_REACTIONS + num_waves + NUM_REACTIONS + (SIZE + 1))
        + 4 * NUM_REACTIONS * sizeof(unsigned char)
        + (2 * (SIZE + 1) + NUM_FLUXES_PLUS + NUM_FLUXES_MINUS) * sizeof(unsigned short)
        + (NUM_FLUXES_PLUS + NUM_FLUXES_MINUS) * sizeof(unsigned char)
        + SIZE * sizeof(double) + 8;

    hyperion_burner_dev_kernel<<<griddim, blockdim, sharedmem_allocation>>>(
	zones,
        args.temp, args.dens, args.xin, args.xout, args.sdotrate,
	args.prefactor, args.p_0, args.p_1, args.p_2, args.p_3,
	args.p_4, args.p_5, args.p_6, args.aa, args.q_value,
	args.reactant_1, args.reactant_2, args.reactant_3,
	args.f_plus_map, args.f_minus_map, args.f_plus_factor,
	args.f_minus_factor, args.f_plus_max, args.f_minus_max,
	args.num_react_species, args.real_vals, d_rate,
	args.species_perm, phase1_count
    );

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess)
        printf("HIP error: %s\n", hipGetErrorString(err));

    // Copy results back to host
    hipMemcpy(xout, args.xout, zones * num_species * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(sdotrate, args.sdotrate, zones * sizeof(double), hipMemcpyDeviceToHost);

}
// -----------------------------------------------------------------------------
// Free all allocated device memory
// -----------------------------------------------------------------------------
void hip_killall_device_ptrs() {
    #define HIP_FREE(ptr) if(ptr){ hipFree(ptr); ptr = NULL; }

    HIP_FREE(args.temp)
    HIP_FREE(args.dens)
    HIP_FREE(args.xin)
    HIP_FREE(args.xout)
    HIP_FREE(args.sdotrate)
    HIP_FREE(args.burned_zone)
    HIP_FREE(args.prefactor)
    HIP_FREE(args.p_0)
    HIP_FREE(args.p_1)
    HIP_FREE(args.p_2)
    HIP_FREE(args.p_3)
    HIP_FREE(args.p_4)
    HIP_FREE(args.p_5)
    HIP_FREE(args.p_6)
    HIP_FREE(args.aa)
    HIP_FREE(args.q_value)
    HIP_FREE(args.reactant_1)
    HIP_FREE(args.reactant_2)
    HIP_FREE(args.reactant_3)
    HIP_FREE(args.f_plus_map)
    HIP_FREE(args.f_minus_map)
    HIP_FREE(args.f_plus_factor)
    HIP_FREE(args.f_minus_factor)
    HIP_FREE(args.f_plus_max)
    HIP_FREE(args.f_minus_max)
    HIP_FREE(args.num_react_species)
    HIP_FREE(args.int_vals)
    HIP_FREE(args.real_vals)
    HIP_FREE(args.species_perm)
    HIP_FREE(d_rate)

    #undef HIP_FREE

}

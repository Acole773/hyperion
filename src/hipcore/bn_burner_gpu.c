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
    HIP_ALLOC_COPY(args.f_plus_max, f_plus_max, num_species);
    HIP_ALLOC_COPY(args.f_minus_max, f_minus_max, num_species);
    HIP_ALLOC_COPY(args.num_react_species, num_react_species, num_reactions);

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
    dim3 blockdim(256, 1, 1);

    int blocks = 1024; //we can tune this to GPU specs! 
    dim3 griddim(blocks, 1, 1);
    size_t sharedmem_allocation =
	sizeof(double) * blockdim.x; //ONLY reduction buffer 

    hyperion_burner_dev_kernel<<<griddim, blockdim, sharedmem_allocation>>>(
	zones,
	args.temp, args.dens, args.xin, args.xout, args.sdotrate,
	args.prefactor, args.p_0, args.p_1, args.p_2, args.p_3,
	args.p_4, args.p_5, args.p_6, args.aa, args.q_value,
	args.reactant_1, args.reactant_2, args.reactant_3,
	args.f_plus_map, args.f_minus_map, args.f_plus_factor,
	args.f_minus_factor, args.f_plus_max, args.f_minus_max,
	args.num_react_species, args.real_vals, d_rate
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
    HIP_FREE(d_rate)

    #undef HIP_FREE

}

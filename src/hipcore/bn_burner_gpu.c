#include "bn_burner_gpu.h"

#include "../core/store.h"

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

// So that my LSP stops bugging me.
// #ifndef SIZE
// #define SIZE 16
// #endif

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

// We can ONLY allocate these values because they are different per-cell.
static void hyperion_burner_kernel(double* tstep, double* temp, double* dens,
                                   double* xin, double* xout, double* sdotrate,
                                   int zones);

void hyperion_burner_(double* tstep, double* temp, double* dens, double* xin,
                      double* restrict xout, double* sdotrate,
                      uchar* burned_zone, int* size) {

    // hipMemcpy(args[BURNED_ZONE], 8);

    // TODO: A memory limit needs to be set here, the GPU can only hold so much info... after that it needs to be broken into chunks.

    hipMemcpy(args[TEMP], temp, *size * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(args[DENS], dens, *size * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(args[XIN], xin, *size * num_species * sizeof(double),
              hipMemcpyHostToDevice);
    hipMemcpy(args[XOUT], xout, *size * num_species * sizeof(double),
              hipMemcpyHostToDevice);
    hipMemcpy(args[SDOTRATE], sdotrate, *size * sizeof(double),
              hipMemcpyHostToDevice);

    // "1" is how many vals there are (lazy bastard)
    hipMemcpy(args[REAL_VALS], tstep, *size * 1 * sizeof(double),
              hipMemcpyHostToDevice);
    // hipMemcpy(args[INT_VALS], NULL, *size * 1 * sizeof(int),
    //           hipMemcpyHostToDevice);

    // Passing NULL because they are not actually used...
    hyperion_burner_kernel(NULL, NULL, NULL, NULL, NULL, NULL, *size);

    hipMemcpy(xout, args[XOUT], *size * num_species * sizeof(double),
              hipMemcpyDeviceToHost);
    hipMemcpy(sdotrate, args[SDOTRATE], *size * sizeof(double),
              hipMemcpyDeviceToHost);
}

int device_init(int zones) {

    int error = 0;

    // We can allocate AND set memory to these values because they are the same
    // for all cells.
    error += hipMalloc(&args[PREFACTOR], num_reactions * sizeof(double));
    error += hipMemcpy(args[PREFACTOR], prefactor,
                       num_reactions * sizeof(double), hipMemcpyHostToDevice);

    error += hipMalloc(&args[P_0], num_reactions * sizeof(double));
    error += hipMemcpy(args[P_0], p_0, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args[P_1], num_reactions * sizeof(double));
    error += hipMemcpy(args[P_1], p_1, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args[P_2], num_reactions * sizeof(double));
    error += hipMemcpy(args[P_2], p_2, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args[P_3], num_reactions * sizeof(double));
    error += hipMemcpy(args[P_3], p_3, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args[P_4], num_reactions * sizeof(double));
    error += hipMemcpy(args[P_4], p_4, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args[P_5], num_reactions * sizeof(double));
    error += hipMemcpy(args[P_5], p_5, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args[P_6], num_reactions * sizeof(double));
    error += hipMemcpy(args[P_6], p_6, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args[AA], num_species * sizeof(double));
    error += hipMemcpy(args[AA], aa, num_species * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args[Q_VALUE], num_reactions * sizeof(double));
    error += hipMemcpy(args[Q_VALUE], q_value, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args[REACTANT_1], num_reactions * sizeof(int));
    error += hipMemcpy(args[REACTANT_1], reactant_1,
                       num_reactions * sizeof(int), hipMemcpyHostToDevice);

    error += hipMalloc(&args[REACTANT_2], num_reactions * sizeof(int));
    error += hipMemcpy(args[REACTANT_2], reactant_2,
                       num_reactions * sizeof(int), hipMemcpyHostToDevice);

    error += hipMalloc(&args[REACTANT_3], num_reactions * sizeof(int));
    error += hipMemcpy(args[REACTANT_3], reactant_3,
                       num_reactions * sizeof(int), hipMemcpyHostToDevice);

    error += hipMalloc(&args[F_PLUS_MAP], f_plus_total * sizeof(int));
    error += hipMemcpy(args[F_PLUS_MAP], f_plus_map, f_plus_total * sizeof(int),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args[F_MINUS_MAP], f_minus_total * sizeof(int));
    error += hipMemcpy(args[F_MINUS_MAP], f_minus_map,
                       f_minus_total * sizeof(int), hipMemcpyHostToDevice);

    error += hipMalloc(&args[F_PLUS_FACTOR], f_plus_total * sizeof(double));
    error += hipMemcpy(args[F_PLUS_FACTOR], f_plus_factor,
                       f_plus_total * sizeof(double), hipMemcpyHostToDevice);

    error += hipMalloc(&args[F_MINUS_FACTOR], f_minus_total * sizeof(double));
    error += hipMemcpy(args[F_MINUS_FACTOR], f_minus_factor,
                       f_minus_total * sizeof(double), hipMemcpyHostToDevice);

    error += hipMalloc(&args[F_PLUS_MAX], num_species * sizeof(int));
    error += hipMemcpy(args[F_PLUS_MAX], f_plus_max, num_species * sizeof(int),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args[F_MINUS_MAX], num_species * sizeof(int));
    error += hipMemcpy(args[F_MINUS_MAX], f_minus_max,
                       num_species * sizeof(int), hipMemcpyHostToDevice);

    error += hipMalloc(&args[NUM_REACT_SPECIES], num_reactions * sizeof(int));
    error += hipMemcpy(args[NUM_REACT_SPECIES], num_react_species,
                       num_reactions * sizeof(int), hipMemcpyHostToDevice);

    error += hipMalloc(&args[BURNED_ZONE], 8);
    error += hipMalloc(&args[TEMP], zones * sizeof(double));
    error += hipMalloc(&args[DENS], zones * sizeof(double));
    error += hipMalloc(&args[XIN], zones * num_species * sizeof(double));
    error += hipMalloc(&args[XOUT], zones * num_species * sizeof(double));
    error += hipMalloc(&args[SDOTRATE], zones * sizeof(double));

    // "1" is how many vals there are
    error += hipMalloc(&args[INT_VALS], zones * 1 * sizeof(int));
    error += hipMalloc(&args[REAL_VALS], zones * 1 * sizeof(double));

    if (error > 0) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static void hyperion_burner_kernel(double* tstep, double* temp, double* dens,
                                   double* xin, double* xout, double* sdotrate,
                                   int zones) {
    int error;

    struct dim3 blockdim = {256, 1, 1};  // Number of threads
    struct dim3 griddim = {zones, 1, 1}; // Number of blocks
    // TODO: this is finding the total memory used by `__shared__` memory in the
    // kernel, but that is less than obvious and should be made more clear.
    // int sharedmem_allocation = sizeof(double) * (num_reactions + num_reactions +
    //                                              num_species + num_species);
    int sharedmem_allocation = 0;
    if ((error = hipConfigureCall(griddim, blockdim, sharedmem_allocation,
                                  hipStreamDefault)) != hipSuccess) {
        return;
    }

    // TODO: this should be the exact number of args, not 100. (I'm a lazy
    // bastard)
    void** trueargs = malloc(100 * sizeof(void*));
    for (int i = 0; i < 100; i++) {
        trueargs[i] = &args[i];
    }

    // TODO: why doesn't this matter for execution, only calling... wtf?
    // sharedmem_allocation = 0;

    // printf("%i\n", sharedmem_allocation);

    error = hipLaunchKernel(hyperion_burner_dev_kernel, griddim, blockdim,
                            trueargs, sharedmem_allocation, hipStreamDefault);

    printf("%i: %s\n", error, hipGetErrorString(error));

    return;
}

void _killall_ptrs_hipdev() {
    int error = 0;
    error += hipFree(args[TEMP]);
    error += hipFree(args[DENS]);
    error += hipFree(args[XIN]);
    error += hipFree(args[XOUT]);
    error += hipFree(args[SDOTRATE]);
    error += hipFree(args[BURNED_ZONE]);
    error += hipFree(args[PREFACTOR]);
    error += hipFree(args[P_0]);
    error += hipFree(args[P_1]);
    error += hipFree(args[P_2]);
    error += hipFree(args[P_3]);
    error += hipFree(args[P_4]);
    error += hipFree(args[P_5]);
    error += hipFree(args[P_6]);
    error += hipFree(args[AA]);
    error += hipFree(args[Q_VALUE]);
    error += hipFree(args[REACTANT_1]);
    error += hipFree(args[REACTANT_2]);
    error += hipFree(args[REACTANT_3]);
    error += hipFree(args[F_PLUS_MAP]);
    error += hipFree(args[F_MINUS_MAP]);
    error += hipFree(args[F_PLUS_FACTOR]);
    error += hipFree(args[F_MINUS_FACTOR]);
    error += hipFree(args[F_PLUS_MAX]);
    error += hipFree(args[F_MINUS_MAX]);
    error += hipFree(args[NUM_REACT_SPECIES]);
    error += hipFree(args[INT_VALS]);
    error += hipFree(args[REAL_VALS]);
    printf("%i: %s\n", error, hipGetErrorString(error));
    return;
}

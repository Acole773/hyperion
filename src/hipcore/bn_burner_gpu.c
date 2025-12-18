#include "bn_burner_gpu.h"

#include "../core/store.h"
#include "../core/restrict.h"

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

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

burner_args_t args;

// We can ONLY allocate these values because they are different per-cell.
static void hyperion_burner_kernel(double* tstep, double* temp, double* dens,
                                   double* xin, double* xout, double* sdotrate,
                                   int zones);

void hyperion_burner_(double* tstep, double* temp, double* dens, double* xin,
                      double* HYP_RESTRICT xout, double* sdotrate,
                      uchar* burned_zone, int* size) {

    for (int i = 0; i < *size; i++) {
        hyperion_burner_kernel(tstep, &temp[i], &dens[i], xin + (SIZE * i),
        xout + (SIZE * i), &sdotrate[i], *size);
    }
}

int device_init(int zones) {

    int error = 0;

    // We can allocate AND set memory to these values because they are the same
    // for all cells.
    error += hipMalloc(&args.prefactor, num_reactions * sizeof(double));
    error += hipMemcpy(args.prefactor, prefactor,
                       num_reactions * sizeof(double), hipMemcpyHostToDevice);

    error += hipMalloc(&args.p_0, num_reactions * sizeof(double));
    error += hipMemcpy(args.p_0, p_0, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args.p_1, num_reactions * sizeof(double));
    error += hipMemcpy(args.p_1, p_1, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args.p_2, num_reactions * sizeof(double));
    error += hipMemcpy(args.p_2, p_2, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args.p_3, num_reactions * sizeof(double));
    error += hipMemcpy(args.p_3, p_3, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args.p_4, num_reactions * sizeof(double));
    error += hipMemcpy(args.p_4, p_4, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args.p_5, num_reactions * sizeof(double));
    error += hipMemcpy(args.p_5, p_5, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args.p_6, num_reactions * sizeof(double));
    error += hipMemcpy(args.p_6, p_6, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args.aa, num_species * sizeof(double));
    error += hipMemcpy(args.aa, aa, num_species * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args.q_value, num_reactions * sizeof(double));
    error += hipMemcpy(args.q_value, q_value, num_reactions * sizeof(double),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args.reactant_1, num_reactions * sizeof(int));
    error += hipMemcpy(args.reactant_1, reactant_1,
                       num_reactions * sizeof(int), hipMemcpyHostToDevice);

    error += hipMalloc(&args.reactant_2, num_reactions * sizeof(int));
    error += hipMemcpy(args.reactant_2, reactant_2,
                       num_reactions * sizeof(int), hipMemcpyHostToDevice);

    error += hipMalloc(&args.reactant_3, num_reactions * sizeof(int));
    error += hipMemcpy(args.reactant_3, reactant_3,
                       num_reactions * sizeof(int), hipMemcpyHostToDevice);

    error += hipMalloc(&args.f_plus_map, f_plus_total * sizeof(int));
    error += hipMemcpy(args.f_plus_map, f_plus_map, f_plus_total * sizeof(int),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args.f_minus_map, f_minus_total * sizeof(int));
    error += hipMemcpy(args.f_minus_map, f_minus_map,
                       f_minus_total * sizeof(int), hipMemcpyHostToDevice);

    error += hipMalloc(&args.f_plus_factor, f_plus_total * sizeof(double));
    error += hipMemcpy(args.f_plus_factor, f_plus_factor,
                       f_plus_total * sizeof(double), hipMemcpyHostToDevice);

    error += hipMalloc(&args.f_minus_factor, f_minus_total * sizeof(double));
    error += hipMemcpy(args.f_minus_factor, f_minus_factor,
                       f_minus_total * sizeof(double), hipMemcpyHostToDevice);

    error += hipMalloc(&args.f_plus_max, num_species * sizeof(int));
    error += hipMemcpy(args.f_plus_max, f_plus_max, num_species * sizeof(int),
                       hipMemcpyHostToDevice);

    error += hipMalloc(&args.f_minus_max, num_species * sizeof(int));
    error += hipMemcpy(args.f_minus_max, f_minus_max,
                       num_species * sizeof(int), hipMemcpyHostToDevice);

    error += hipMalloc(&args.num_react_species, num_reactions * sizeof(int));
    error += hipMemcpy(args.num_react_species, num_react_species,
                       num_reactions * sizeof(int), hipMemcpyHostToDevice);

    error += hipMalloc(&args.burned_zone, 8);
    error += hipMalloc(&args.temp, zones * sizeof(double));
    error += hipMalloc(&args.dens, zones * sizeof(double));
    error += hipMalloc(&args.xin, zones * num_species * sizeof(double));
    error += hipMalloc(&args.xout, zones * num_species * sizeof(double));
    error += hipMalloc(&args.sdotrate, zones * sizeof(double));

    // "1" is how many vals there are
    error += hipMalloc(&args.int_vals, zones * 1 * sizeof(int));
    error += hipMalloc(&args.real_vals, zones * 1 * sizeof(double));

    if (error > 0) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static void hyperion_burner_kernel(double* tstep, double* temp, double* dens,
                                   double* xin, double* xout, double* sdotrate,
                                   int zones) {
    int error;
    burner_args_t args;

    // hipMemcpy(args[BURNED_ZONE], 8);
    hipMemcpy(args.temp, temp, zones * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(args.dens, dens, zones * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(args.xin, xin, zones * num_species * sizeof(double),
              hipMemcpyHostToDevice);
    hipMemcpy(args.xout, xout, zones * num_species * sizeof(double),
              hipMemcpyHostToDevice);
    hipMemcpy(args.sdotrate, sdotrate, zones * sizeof(double),
              hipMemcpyHostToDevice);

    // "1" is how many vals there are
    hipMemcpy(args.real_vals, tstep, zones * 1 * sizeof(double),
              hipMemcpyHostToDevice);
    // hipMemcpy(args[INT_VALS], NULL, zones * 1 * sizeof(int),
    //           hipMemcpyHostToDevice);

    struct dim3 blockdim = {256, 1, 1}; // Number of threads
    struct dim3 griddim = {1, 1, 1};    // Number of blocks
    // TODO: this is finding the total memory used by `__shared__` memory in the
    // kernel, but that is less than obvious and should be made more clear.
    size_t sharedmem_allocation =
        sizeof(double) * (num_reactions + num_reactions +
                      f_plus_total + f_minus_total +
                      num_species + num_species);

    args.temp = temp;
    args.dens = dens;
    args.xin  = xin;
    args.xout = xout;
    args.sdotrate = sdotrate;

    args.prefactor = prefactor;
    args.p_0 = p0;
    args.p_1 = p1;
    args.p_2 = p2;
    args.p_3 = p3;
    args.p_4 = p4;
    args.p_5 = p5;
    args.p_6 = p6;
    args.aa = aa;
    args.q_value = q_value;

    args.reactant_1 = reactant_1;
    args.reactant_2 = reactant_2;
    args.reactant_3 = reactant_3;

    args.f_plus_map = f_plus_map;

    args.f_minus_map = f_minus_map;
    args.f_plus_factor = f_plus_factor;
    args.f_minus_factor = f_minus_factor;

    args.f_plus_max = f_plus_max;
    args.f_minus_max = f_minus_max;

    args.num_react_species = num_react_species;

    args.real_vals = real_vals;

    hyperion_burner_dev_kernel<<<griddim, blockdim, sharedmem_allocation>>>(
        args.temp,
        args.dens,
        args.xin,
        args.xout,
        args.sdotrate,
        args.prefactor,
	args.p_0,
	args.p_1,
	args.p_2,
	args.p_3,
	args.p_4,
	args.p_5,
	args.p_6,
    	args.aa,
    	args.q_value,
	args.reactant_1,
    	args.reactant_2,
    	args.reactant_3,
    	args.f_plus_map,
    	args.f_minus_map,
    	args.f_plus_factor,
    	args.f_minus_factor,
    	args.f_plus_max,
    	args.f_minus_max,
    	args.num_react_species,
    	args.real_vals
    );

    printf("%i: %s\n", error, hipGetErrorString(error));

    hipMemcpy(xout, args.xout, zones * num_species * sizeof(double),
              hipMemcpyDeviceToHost);
    hipMemcpy(sdotrate, args.sdotrate, zones * sizeof(double),
              hipMemcpyDeviceToHost);

    return;
}

void _killall_ptrs_hipdev() {
    int error = 0;
    error += hipFree(args.temp);
    error += hipFree(args.dens);
    error += hipFree(args.xin);
    error += hipFree(args.xout);
    error += hipFree(args.sdotrate);
    error += hipFree(args.burned_zone);
    error += hipFree(args.prefactor);
    error += hipFree(args.p_0);
    error += hipFree(args.p_1);
    error += hipFree(args.p_2);
    error += hipFree(args.p_3);
    error += hipFree(args.p_4);
    error += hipFree(args.p_5);
    error += hipFree(args.p_6);
    error += hipFree(args.aa);
    error += hipFree(args.q_value);
    error += hipFree(args.reactant_1);
    error += hipFree(args.reactant_2);
    error += hipFree(args.reactant_3);
    error += hipFree(args.f_plus_map);
    error += hipFree(args.f_minus_map);
    error += hipFree(args.f_plus_factor);
    error += hipFree(args.f_minus_factor);
    error += hipFree(args.f_plus_max);
    error += hipFree(args.f_minus_max);
    error += hipFree(args.num_react_species);
    error += hipFree(args.int_vals);
    error += hipFree(args.real_vals);
    printf("%i: %s\n", error, hipGetErrorString(error));
    return;
}

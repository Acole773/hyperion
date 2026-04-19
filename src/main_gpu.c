#include "gpu/gpu_backend.h"
#include "core/paths.h"

#include "core/init.h"
#include "core/kill.h"
#include "core/store.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <x86intrin.h>
#include <hip/hip_runtime.h>

#define BATCHCNT 8 // Number of zones to compute, this will get over written by main arguments. 

int run_batch(int, int);

int main(int argc, char** argv) {
    /*    DEFAULT BEHAVOR    */

    int zones = BATCHCNT;
    int num_iterations = 1;

    /*    OVERWRITE WITH MAIN ARGUMENTS    */

    if (argc > 1) {
        char* end; //It tells you where parsing stopped
	long val = strtol(argv[1], &end, 10); //Same as atoi, but more flexible

	if (*end != '\0' || val <= 0) {  //insure the entire string was a valid integer
	    fprintf(stderr, "Invalid zones argument: %s\n", argv[1]);
	    return EXIT_FAILURE;
	}

	zones = (int)val;
	fprintf(stderr, "Using zones from CLI: %d\n", zones);
    } else {
	fprintf(stderr, "Using default zones: %d\n", zones);
    }

    if (argc > 2) {
	char* end;
	long val = strtol(argv[2], &end, 10);
	if (*end != '\0' || val <= 0) {
	    fprintf(stderr, "Invalid iterations argument: %s\n", argv[2]);
	    return EXIT_FAILURE;
	}
	num_iterations = (int)val;
	fprintf(stderr, "Using iterations from CLI: %d\n", num_iterations);
    } else {
	fprintf(stderr, "Using default iterations: %d\n", num_iterations);
    }
    fflush(stderr);

    hyperion_data_dir = getenv("HYPERION_DATA_DIR");
    if (!hyperion_data_dir) {
        fprintf(stderr,
            "ERROR: HYPERION_DATA_DIR not set\n"
            "Please export HYPERION_DATA_DIR=/path/to/hyperion/data\n");
        return EXIT_FAILURE;
    }

    hyperion_init_(); // build network on host
    if (gpu_backend_init(zones) == EXIT_FAILURE) {
	fprintf(stderr, "GPU backend init failed\n");
        return EXIT_FAILURE;
    }
    if (run_batch(zones, num_iterations) == EXIT_FAILURE) {
	fprintf(stderr, "run_batch failed\n");
        return EXIT_FAILURE;
    }

    gpu_backend_finalize();

    return EXIT_SUCCESS;
}

int run_batch(int zones, int num_iterations) {
    int size = SIZE;

    double tstep = 1e-06;
    unsigned char* burned_zone = malloc(zones * sizeof(unsigned char));
    memset(burned_zone, 0, zones);

    int* kstep;

    double* _scope_xin = malloc((size * zones) * sizeof(double) + 0x40);
    double* _scope_xout = malloc((size * zones) * sizeof(double) + 0x40);
    double* _scope_sdotrate = malloc(zones * sizeof(double) + 0x40);
    double* xin = (double*)(((uintptr_t)_scope_xin) + 0x3F & ~0x3F);
    double* xout = (double*)(((uintptr_t)_scope_xout) + 0x3F & ~0x3F);
    double* sdotrate = (double*)(((uintptr_t)_scope_sdotrate) + 0x3F & ~0x3F);

    double* temp = malloc(zones * sizeof(double));
    double* dens = malloc(zones * sizeof(double));

    // =========================
    // Timing variables
    // =========================
    struct timespec start, end;
    double wall_time;

    // GPU timing
    hipEvent_t gpu_start, gpu_stop;
    float gpu_time_ms;   // HIP returns milliseconds
    double gpu_time;     // convert to seconds

    // =========================
    // Setup GPU timers
    // =========================
    hipEventCreate(&gpu_start);
    hipEventCreate(&gpu_stop);

    /**********************************
     *
     * *******************************/

    for (int i = 0; i < zones; i++) {
	double* current_xin = xin + (size *i);
        memset(current_xin, 0, size * sizeof(double));
	current_xin[12] = 0.04166;
	current_xin[20] = 0.03125;
        temp[i] = 5e09;
        dens[i] = 1e08;
    }
     


/*
    srand((unsigned int)time(NULL));

    for (int i = 0; i < zones; i++) {
        memcpy(xin + (size * i), x, size * sizeof(double));

        // random factor between 1/1.5 and 1.5 
        double r = (double)rand() / (double)RAND_MAX;
        double factor_T = (1.0/1.5) + r * (1.5 - 1.0/1.5);

        r = (double)rand() / (double)RAND_MAX;
        double factor_D = (1.0/1.5) + r * (1.5 - 1.0/1.5);

        temp[i] = 5e09 * factor_T;
        dens[i] = 1e08 * factor_D;
     }
*/
    

/*--------------------------------------------------------------
 *  GAUSSIAN PERTURBATION SCHEME FOR INITIAL CONDITIONS
 *
 *  Generates Gaussian aka normal perturbations for temperature
 *  and density.
 *
 *      T_i   = 5e09 * (1 + delta_T)
 *      rho_i = 1e08 * (1 + delta_D)
 *
 *  delta_T, delta_D ~ N(0, sigma), truncated to ±3σ.
 *
 *  Not sure if the truncation I did will cause a bunch-up at 3 sigma.
 *--------------------------------------------------------------*/
/*
    srand((unsigned int)time(NULL));

    // variable definitions grouped together
    double sigma_T = 0.10;
    double sigma_D = 0.10;

    double u1, u2;
    double z1, z2;
    double delta_T, delta_D;

    for (int i = 0; i < zones; i++) {
        memcpy(xin + (size * i), x, size * sizeof(double));

    // Gaussian sample for temperature
        u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        z1 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    // Gaussian sample for density
        u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        z2 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

    // scaling
        delta_T = sigma_T * z1;
        delta_D = sigma_D * z2;

    // truncate at ±3σ
        if (delta_T > 3*sigma_T) delta_T = 3*sigma_T;
        if (delta_T < -3*sigma_T) delta_T = -3*sigma_T;

        if (delta_D > 3*sigma_D) delta_D = 3*sigma_D;
        if (delta_D < -3*sigma_D) delta_D = -3*sigma_D;

        temp[i] = 5e09 * (1.0 + delta_T);
        dens[i] = 1e08 * (1.0 + delta_D);
    }

    
*/
    /**********************************
     *
     * *******************************/

    //NEED to INITIALIZE DEVICE HERE
    
    // WARMUP

    int wrm_zones = 8; 
    gpu_burner(&tstep, temp, dens, xin, xout, sdotrate, burned_zone,
                     &wrm_zones);

    unsigned long long cycles = __rdtsc();

    // =========================
    // Start timers
    // =========================
    clock_gettime(CLOCK_MONOTONIC, &start);
    hipEventRecord(gpu_start, 0);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        gpu_burner(&tstep, temp, dens, xin, xout, sdotrate, burned_zone,
                         &zones);
    }

    // =========================
    // Stop timers
    // =========================
    hipEventRecord(gpu_stop, 0);
    hipEventSynchronize(gpu_stop);

    clock_gettime(CLOCK_MONOTONIC, &end);

    // =========================
    // Compute elapsed times
    // =========================

    // Wall clock (seconds)
    wall_time =
	(end.tv_sec - start.tv_sec) +
	(end.tv_nsec - start.tv_nsec) * 1e-9;

    // GPU time (convert ms → seconds)
    hipEventElapsedTime(&gpu_time_ms, gpu_start, gpu_stop);
    gpu_time = gpu_time_ms * 1e-3;

    // =========================
    // Cleanup (important at scale)
    // =========================
    hipEventDestroy(gpu_start);
    hipEventDestroy(gpu_stop);

    double avg_wall_time = wall_time / num_iterations;
    double avg_gpu_time = gpu_time / num_iterations;

    unsigned long long cycles_ = __rdtsc();

    printf("iterations, zones, avg_wall_time, avg_gpu_time, total_wall_time, total_gpu_time, cycles \n");

    printf("%d %d %f %f %f %f %llu\n", num_iterations, zones,
           avg_wall_time, avg_gpu_time, wall_time, gpu_time, cycles_);

    printf( "END OF TABLE\n");

    printf("Result (from last iteration):\n");

    for (int i = 0; i < size; i++) {
        printf("%4i %.5e\n", i, xout[i]);
    }
    printf("\n");

    printf("Sdotrate for the batch (from last iteration).\n");
    for (int i = 0; i < zones; i++) {
        printf("sdot[%i]: %.5e\n", i, sdotrate[i]);
    }

    printf("Total cycles per run of batch (avg over %d iterations, rnded): %lld \n",
           num_iterations, (cycles_ - cycles) / (zones * num_iterations));

    free(burned_zone);
    free(_scope_xin);
    free(_scope_xout);
    free(_scope_sdotrate);

    return EXIT_SUCCESS;
}

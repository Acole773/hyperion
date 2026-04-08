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

#define BATCHCNT 8 // Number of zones to compute

int run_batch(void);

int main(int argc, char** argv) {
    /*    DEFAULT BEHAVOR    */

    int zones = BATCHCNT;

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
    if (run_batch() == EXIT_FAILURE) {
	fprintf(stderr, "run_batch failed\n");
        return EXIT_FAILURE;
    }

    gpu_backend_finalize();

    return EXIT_SUCCESS;
}

int run_batch(void) {
    int size = SIZE;

    double tstep = 1e-06;
    unsigned char* burned_zone = malloc(BATCHCNT * sizeof(unsigned char));
    memset(burned_zone, 0, BATCHCNT);
    int* zone;
    int* kstep;

    double* _scope_xin = malloc((size * BATCHCNT) * sizeof(double) + 0x40);
    double* _scope_xout = malloc((size * BATCHCNT) * sizeof(double) + 0x40);
    double* _scope_sdotrate = malloc(BATCHCNT * sizeof(double) + 0x40);
    double* xin = (double*)(((uintptr_t)_scope_xin) + 0x3F & ~0x3F);
    double* xout = (double*)(((uintptr_t)_scope_xout) + 0x3F & ~0x3F);
    double* sdotrate = (double*)(((uintptr_t)_scope_sdotrate) + 0x3F & ~0x3F);

    double* temp = malloc(BATCHCNT * sizeof(double));
    double* dens = malloc(BATCHCNT * sizeof(double));

    /**********************************
     *
     * *******************************/

    for (int i = 0; i < BATCHCNT; i++) {
	double* current_xin = xin + (size *i);
        memset(current_xin, 0, size * sizeof(double));
	current_xin[12] = 0.04166;
	current_xin[20] = 0.03125;
        temp[i] = 5e09;
        dens[i] = 1e08;
    }
     


/*
    srand((unsigned int)time(NULL));

    for (int i = 0; i < BATCHCNT; i++) {
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

    for (int i = 0; i < BATCHCNT; i++) {
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


    int zones = BATCHCNT;
    //NEED to INITIALIZE DEVICE HERE
    
    // WARMUP
    gpu_burner(&tstep, temp, dens, xin, xout, sdotrate, burned_zone,
                     &zones);

    unsigned long long cycles = __rdtsc();

    gpu_burner(&tstep, temp, dens, xin, xout, sdotrate, burned_zone,
                     &zones);

    unsigned long long cycles_ = __rdtsc();

    printf("Result:\n");

    for (int i = 0; i < size; i++) {
        printf("%4i %.5e\n", i, xout[i]);
    }
    printf("\n");

    printf("Sdotrate for the batch.\n");
    for (int i = 0; i < BATCHCNT; i++) {
        printf("sdot[%i]: %.5e\n", i, sdotrate[i]);
    }

    printf("Total cycles per run of batch (avg, rnded): %lld \n",
           (cycles_ - cycles) / BATCHCNT);

    free(burned_zone);
    free(_scope_xin);
    free(_scope_xout);
    free(_scope_sdotrate);

    return EXIT_SUCCESS;
}

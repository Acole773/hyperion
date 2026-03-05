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
    fprintf(stderr, ">>> ENTERED MAIN <<<\n");
    fflush(stderr);
    fprintf(stderr, "MAIN: before gpu_backend_init\n");
    fflush(stderr);
    int zones = BATCHCNT;
    hyperion_data_dir = getenv("HYPERION_DATA_DIR");
    if (!hyperion_data_dir) {
        fprintf(stderr,
            "ERROR: HYPERION_DATA_DIR not set\n"
            "Please export HYPERION_DATA_DIR=/path/to/hyperion/data\n");
        return EXIT_FAILURE;
    }

    printf("Initializing Hyperion...\n");
    hyperion_init_(); // build network on host
    printf("Hyperion initialized.\n");
    printf("Starting GPU backend init...\n");
    if (gpu_backend_init(zones) == EXIT_FAILURE) {
	fprintf(stderr, "GPU backend init failed\n");
        return EXIT_FAILURE;
    }
    printf("GPU backend init done.\n");
    printf("Starting run_batch...\n");
    if (run_batch() == EXIT_FAILURE) {
	fprintf(stderr, "run_batch failed\n");
        return EXIT_FAILURE;
    }
    printf("run_batch done.\n");

    printf("Finalizing GPU backend...\n");
    gpu_backend_finalize();
    printf("GPU backend finalize done.\n");

    return EXIT_SUCCESS;
}

int run_batch(void) {
    int size = SIZE;

    double tstep = 1e-06;
    unsigned char* burned_zone = malloc(BATCHCNT * sizeof(unsigned char));
    memset(burned_zone, 0, BATCHCNT);
    int* zone;
    int* kstep;

    printf("Allocating host arrays...\n");
    double* _scope_xin = malloc((size * BATCHCNT) * sizeof(double) + 0x40);
    double* _scope_xout = malloc((size * BATCHCNT) * sizeof(double) + 0x40);
    double* _scope_sdotrate = malloc(BATCHCNT * sizeof(double) + 0x40);
    double* xin = (double*)(((uintptr_t)_scope_xin) + 0x3F & ~0x3F);
    double* xout = (double*)(((uintptr_t)_scope_xout) + 0x3F & ~0x3F);
    double* sdotrate = (double*)(((uintptr_t)_scope_sdotrate) + 0x3F & ~0x3F);

    double* temp = malloc(BATCHCNT * sizeof(double));
    double* dens = malloc(BATCHCNT * sizeof(double));
    printf("Host arrays allocated.\n");

    /**********************************
     *
     * *******************************/

    printf("Copying initial conditions into xin...\n");
    for (int i = 0; i < BATCHCNT; i++) {
        memcpy(xin + (size * i), x, size * sizeof(double));
        temp[i] = 5e09;
        dens[i] = 1e08;
    }
    printf("Initial conditions copied.\n");
    


    /*printf("Copying initial conditions into xin...\n");
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
    printf("Initial conditions copied.\n");
*/
    

    /**********************************
     *
     * *******************************/


    int zones = BATCHCNT;
    //NEED to INITIALIZE DEVICE HERE
    
    // WARMUP
    printf("Launching GPU burner kernel (warmup)...\n");
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

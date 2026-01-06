#include "gpu/gpu_backend.h"

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

int main() {

    if (gpu_backend_init(BATCHCNT) == EXIT_FAILURE) {
        return EXIT_FAILURE;
    }

    if (run_batch() == EXIT_FAILURE) {
        return EXIT_FAILURE;
    }

    gpu_backend_finalize();
    return EXIT_SUCCESS;
}

int run_batch(void) {
    int size = SIZE;

    double tstep = 1e-06;
    unsigned char* burned_zone;
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

    hyperion_init_();

    for (int i = 0; i < BATCHCNT; i++) {
        memcpy(xin + (size * i), x, size * sizeof(double));
        temp[i] = 5e09;
        dens[i] = 1e08;
    }

    int zones = BATCHCNT;

    if (gpu_backend_init(zones) != EXIT_SUCCESS) {
        fprintf(stderr, "GPU backend init failed\n");
        return EXIT_FAILURE;
    }
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

    free(_scope_xin);
    free(_scope_xout);
    free(_scope_sdotrate);

    _killall_ptrs();

    return EXIT_SUCCESS;
}

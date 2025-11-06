#include "core/bn_burner_gpu.h"
#include "core/init.h"
#include "core/kill.h"
#include "core/store.h"
#include "gpu/hip.h"
#include "gpu/hip.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <x86intrin.h>

// The choice for how long to warm up is complicated. It just needs to max the
// thread's clock out. There's a lot that goes into this... but this is fine.
// #define WARMUP 4096 * 10
// #define AFRN 256 // "Arbitrary Fucking Run Number"

// For the big one
// #define WARMUP 1024
// #define AFRN 256

// Testing and FEST
#define WARMUP 0
#define AFRN 1

#define BATCHCNT 8

int run_no_batch(void);

int main() {

    struct hipDeviceProp_t* hip_device = get_hip_device();
    if (hip_device == NULL) {
        return EXIT_FAILURE;
    }

    if (test_device(hip_device) == EXIT_FAILURE) {
        return EXIT_FAILURE;
    }

    if (run_no_batch() == EXIT_FAILURE) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int run_no_batch(void) {
    int size = SIZE;

    double sdotrate;
    uchar* burned_zone;

    double* __scope_xin = malloc(size * sizeof(double) + 0x40);
    double* __scope_xout = malloc(size * sizeof(double) + 0x40);
    double* xin = (double*)(((uintptr_t)__scope_xin) + 0x3F & ~0x3F);
    double* xout = (double*)(((uintptr_t)__scope_xout) + 0x3F & ~0x3F);

    double temp = 5e09;
    double dens = 1e08;
    double tstep = 1e-06;

    hyperion_init_();

    memcpy(xin, x, size * sizeof(double));

    // Assemble gpu shit (some of this should go in a )

    // TODO: GPU warmup & check
    args = malloc(100 * sizeof(void*)); // TODO: set this shit up
    if (device_init() == EXIT_FAILURE) {
        return EXIT_FAILURE;
    }
    
    hipError_t error;
    struct dim3 griddim = {1, 1, 1};
    struct dim3 blockdim = {1, 1, 1};
    
    for (int i = 0; i < WARMUP; i++) {
        hyperion_burner_(&tstep, &temp, &dens, xin, xout, &sdotrate,
                         burned_zone, &size);
    }

    // TODO: this timing code is buggy on the best of days...
    unsigned long long cycles = __rdtsc();

    for (int i = 0; i < AFRN; i++) {
        hyperion_burner_(&tstep, &temp, &dens, xin, xout, &sdotrate,
                         burned_zone, &size);
    }

    unsigned long long cycles_ = __rdtsc();

    printf("Total cycles per run (avg, rnded): %lld \n",
           (cycles_ - cycles) / AFRN);

    printf("Result:\n");

    double sum = 0;
    for (int i = 0; i < size; i++) {
        printf("%2i %.5e\n", i, xout[i]);
        sum += xout[i];
    }
    printf("\n");
    printf("sdot: %.5e\n", sdotrate);

    free(__scope_xin);
    free(__scope_xout);

    _killall_ptrs();
    _killall_ptrs_hipdev();
    free(args);

    return EXIT_SUCCESS;
}

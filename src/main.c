#include "core/bn_burner.h"
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

// The choice for how long to warm up is complicated. It just needs to max the
// thread's clock out. There's a lot that goes into this... but this is fine.
// #define WARMUP 4096 * 10
// #define AFRN 256 // "Arbitrary Fucking Run Number"

// For the big one
// #define WARMUP 1024
// #define AFRN 256

// Testing and FEST
// #define WARMUP 0
// #define AFRN 1

#define BATCHCNT 32 // Number of zones to compute

int run_batch(void);

int main() {

    if (run_batch() == EXIT_FAILURE) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int run_batch(void) {
    int size = SIZE;

    double tstep = 1e-06;
    uchar* burned_zone;
    int* zone;
    int* kstep;

    // Overallocate for non-batched, but it's fine since we'll just ignore it if
    // that's the case.
    double* _scope_xin = malloc((size * BATCHCNT) * sizeof(double) + 0x40);
    double* _scope_xout = malloc((size * BATCHCNT) * sizeof(double) + 0x40);
    double* _scope_sdotrate = malloc(BATCHCNT * sizeof(double) + 0x40);
    double* xin      = (double*)((((uintptr_t)_scope_xin      + 0x3F) & ~0x3F));
    double* xout     = (double*)((((uintptr_t)_scope_xout     + 0x3F) & ~0x3F));
    double* sdotrate = (double*)((((uintptr_t)_scope_sdotrate + 0x3F) & ~0x3F));

    double* temp = malloc(BATCHCNT * sizeof(double));
    double* dens = malloc(BATCHCNT * sizeof(double));

    hyperion_init_();

    for (int i = 0; i < BATCHCNT; i++) {
        memcpy(xin + (size * i), x, size * sizeof(double));
        temp[i] = 5e09;
        dens[i] = 1e08;
    }

    int zones = BATCHCNT;
    burned_zone = malloc(zones * sizeof(uchar));

    // WARMUP
    hyperion_burner_(&tstep, temp, dens, xin, xout, sdotrate, burned_zone,
                     &zones);

    unsigned long long cycles = __rdtsc();

#ifdef __HYPERION_USE_SIMD
    hyperion_burner_(&tstep, temp, dens, xin, xout, sdotrate, burned_zone,
                     &zones);
#else
    hyperion_burner_(&tstep, temp, dens, xin, xout, sdotrate, burned_zone,
                     &zones);
#endif

    unsigned long long cycles_ = __rdtsc();

    printf("Result:\n");

#ifdef __HYPERION_USE_SIMD
    // Add size just to check that it's working (serial cannot do this..)
    // Yes, this is all a little funky, but it works fine if you know what's
    // going on.
    // for (int i = 0; i < size; i++) {
    //     printf("%2i %.5e\n", i, xout[i]);
    // }
#else
    // for (int i = 0; i < size; i++) {
    //     printf("%4i %.5e\n", i, xout[i]);
    // }
#endif
    printf("\n");

    // printf("Sdotrate for the batch.\n");
    // for (int i = 0; i < BATCHCNT; i++) {
    //     printf("sdot[%i]: %.5e\n", i, sdotrate[i]);
    // }

    printf("Total cycles per run of batch (avg, rnded): %lld \n",
           (cycles_ - cycles) / BATCHCNT);

    free(_scope_xin);
    free(_scope_xout);
    free(_scope_sdotrate);

    _killall_ptrs();

    return EXIT_SUCCESS;
}

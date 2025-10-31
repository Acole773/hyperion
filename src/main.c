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

#include "debug/tickertape.h" //AC

// The choice for how long to warm up is complicated. It just needs to max the
// thread's clock out. There's a lot that goes into this... but this is fine.
// #define WARMUP 4096 * 10
// #define AFRN 256 // "Arbitrary Fucking Run Number"

// For the big one
#define WARMUP 1024
#define AFRN 256

// Testing and FEST
// #define WARMUP 0
// #define AFRN 1

#define BATCHCNT 8

void run_no_batch();
void run_with_batch();

int main() {
    tickertape_init("/home/adam/hyperion/debug_log.txt");

    tickertape_log("Tickertape initialized.");


    // run_no_batch(); // NOT COMPAT WITH SIMD
    run_with_batch();

    tickertape_close();

    return EXIT_SUCCESS;
}

void run_no_batch() {
    // TODO: this is bad
    int size = SIZE;

    double sdotrate;
    uchar* burned_zone;

    // I'm positive there's a more elegant way to do this, but we are aligning
    // our pointers to a 64 byte boundary (assuming that Flash-X will afford us
    // the same kindness. Within our own code, we certainly will).
    double* __scope_xin = malloc(size * sizeof(double) + 0x40);
    double* __scope_xout = malloc(size * sizeof(double) + 0x40);
    double* xin = (double*)(((uintptr_t)__scope_xin) + 0x3F & ~0x3F);
    double* xout = (double*)(((uintptr_t)__scope_xout) + 0x3F & ~0x3F);

    double temp = 5e09;
    double dens = 1e08;
    double tstep = 1e-06;

    hyperion_init_();

    tickertape_log("This is a test.");
    tickertape_log("Value of pi ≈ %.5f", 3.14159);

    memcpy(xin, x, size * sizeof(double));

    // Warm up, yes this matters.
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

    return;
}

void run_with_batch() {
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
    double* xin = (double*)(((uintptr_t)_scope_xin) + 0x3F & ~0x3F);
    double* xout = (double*)(((uintptr_t)_scope_xout) + 0x3F & ~0x3F);
    double* sdotrate = (double*)(((uintptr_t)_scope_sdotrate) + 0x3F & ~0x3F);

    // Technically this should be scoped for good rate as well but eeee who cares
    double temp[BATCHCNT] = {5e09,   5.2e09, 5.4e09, 5.6e09,
                             5.8e09, 6e09,   6.2e09, 6.4e09};

    double dens[BATCHCNT] = {1e08,   1.1e08, 1.2e08, 1.3e08,
                             1.4e08, 1.5e08, 1.6e08, 1.7e08};

    hyperion_init_();

    // memcpy(xin, x, size * sizeof(double));

    // The alignment won't always work out like this, so a better method is
    // needed... it just so happens that 16 * 8 = 128, which is divisible by 64,
    // so it works...
    for (int i = 0; i < BATCHCNT; i++) {
        memcpy(xin + (size * i), x, size * sizeof(double));
    }

    // TODO: make the warmup not use SIMD for the SIMD version...
    for (int i = 0; i < WARMUP; i++) {
        hyperion_burner_(&tstep, temp, dens, xin, xout, sdotrate, burned_zone,
                         &size);
    }

    unsigned long long cycles = __rdtsc();

    for (int i = 0; i < AFRN; i++) {
#ifdef __HYPERION_USE_SIMD
        // We are going to make a bunch of assumptions about stride and whatnot
        // that will need to be well documented. For now it'll be in
        // bn_burner_SIMD.c
        // Basically: this is assumed to be BATCHCNT batched
        hyperion_burner_(&tstep, temp, dens, xin, xout, sdotrate, burned_zone,
                         &size);
#else
        for (int j = 0; j < BATCHCNT; j++) {
            hyperion_burner_(&tstep, temp, dens, xin, xout, sdotrate,
                             burned_zone, &size);
        }
#endif
    }

    unsigned long long cycles_ = __rdtsc();

    printf("Total cycles per run of batch (avg, rnded): %lld \n",
           (cycles_ - cycles) / AFRN);

    printf("Result:\n");

    double sum = 0;
#ifdef __HYPERION_USE_SIMD
    // Add size just to check that it's working (serial cannot do this..)
    // Yes, this is all a little funky, but it works fine if you know what's
    // going on.
    for (int i = 0; i < size; i++) {
        printf("%2i %.5e\n", i, xout[i]);
        sum += xout[i];
    }
#else
    for (int i = 0; i < size; i++) {
        printf("%4i %.5e\n", i, xout[i]);
        sum += xout[i];
    }
#endif
    printf("\n");
    
    printf("Sdotrate for the batch.\n");
    for (int i = 0; i < BATCHCNT; i++) {
        printf("sdot: %.5e\n", sdotrate[i]);
        sum += xout[i];
    }

    free(_scope_xin);
    free(_scope_xout);
    free(_scope_sdotrate);

    _killall_ptrs();

    return;
}

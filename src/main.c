#include "core/bn_burner.h"
#include "core/init.h"
#include "core/kill.h"
#include "core/paths.h"
#include "core/store.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

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

#define BATCHCNT 8 // Match the current GPU driver setup

int run_batch(void);
static unsigned long long hyperion_tick_count(void);
static int configure_data_dir(void);
static void setup_like_gpu_problem(double* xin, double* temp, double* dens,
                                   int size);

int main() {
    if (configure_data_dir() == EXIT_FAILURE) {
        return EXIT_FAILURE;
    }

    if (run_batch() == EXIT_FAILURE) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int run_batch(void) {
    int size = SIZE;

    double tstep = 1e-06;
    uchar* burned_zone;

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
    memset(sdotrate, 0, BATCHCNT * sizeof(double));

    printf("Initializing Hyperion...\n");
    hyperion_init_();
    printf("Hyperion initialized.\n");

    printf("Allocating and initializing CPU problem...\n");
    setup_like_gpu_problem(xin, temp, dens, size);
    printf("CPU problem initialized.\n");

    int zones = BATCHCNT;
    burned_zone = malloc(zones * sizeof(uchar));
    memset(burned_zone, 0, zones * sizeof(uchar));

    // WARMUP
    printf("Launching CPU burner (warmup)...\n");
    hyperion_burner_(&tstep, temp, dens, xin, xout, sdotrate, burned_zone,
                     &zones);

    unsigned long long cycles = hyperion_tick_count();

#ifdef __HYPERION_USE_SIMD
    hyperion_burner_(&tstep, temp, dens, xin, xout, sdotrate, burned_zone,
                     &zones);
#else
    hyperion_burner_(&tstep, temp, dens, xin, xout, sdotrate, burned_zone,
                     &zones);
#endif

    unsigned long long cycles_ = hyperion_tick_count();

    // Print abundance Y here so the standalone CPU log matches the current
    // GPU driver's Result block without changing the CPU burner's X-space API.
    printf("Result:\n");
    for (int i = 0; i < size; i++) {
        printf("%4i %.5e\n", i, xout[i] / aa[i]);
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

    _killall_ptrs();

    return EXIT_SUCCESS;
}

static int configure_data_dir(void) {
    const char* env_data_dir = getenv("HYPERION_DATA_DIR");

    if (env_data_dir && env_data_dir[0] != '\0') {
        hyperion_data_dir = env_data_dir;
    }

    if (!hyperion_data_dir) {
        fprintf(stderr,
                "ERROR: HYPERION_DATA_DIR not set\n"
                "Set HYPERION_DATA_DIR=/path/to/hyperion or use a CMake build "
                "that embeds a default data path.\n");
        return EXIT_FAILURE;
    }

    fprintf(stderr, "Using HYPERION_DATA_DIR=%s\n", hyperion_data_dir);
    return EXIT_SUCCESS;
}

static unsigned long long hyperion_tick_count(void) {
#if defined(__x86_64__) || defined(__i386__)
    return __rdtsc();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((unsigned long long)ts.tv_sec * 1000000000ull) +
           (unsigned long long)ts.tv_nsec;
#endif
}

static void setup_like_gpu_problem(double* xin, double* temp, double* dens,
                                   int size) {
    for (int i = 0; i < BATCHCNT; i++) {
        double* current_xin = xin + (size * i);

        memset(current_xin, 0, size * sizeof(double));

        // Mirror the current GPU driver setup so CPU and GPU runs start from
        // the same hard-coded problem definition.
        if (size > 20) {
            current_xin[12] = 0.04166;
            current_xin[20] = 0.03125;
        } else {
            memcpy(current_xin, x, size * sizeof(double));
        }

        temp[i] = 5e09;
        dens[i] = 1e08;
    }
}

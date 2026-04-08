#include "gpu_backend.h"
#include "./hipcore/hip_backend.h"	// declares hip_backend_init, device_init, select_best_device
#include <stdio.h>

#if defined(ENABLE_GPU)

/* ================= HIP BACKEND ================= */

#if defined(USE_HIP)

#include "hipcore/bn_burner_gpu.h"
#include "hipcore/hip_backend.h"

/* Fortran interface */
void hyperion_burner_(double* tstep,
                      double* temp,
                      double* dens,
                      double* xin,
                      double* xout,
                      double* sdotrate,
                      unsigned char* burned_zone,
                      int* zones);

int gpu_backend_init(int zones)
{
    // Select device
    int dev = select_best_device();
    if (dev < 0) {
        fprintf(stderr, "No HIP devices found!\n");
        return EXIT_FAILURE;
    }
    hipError_t err = hipSetDevice(dev);
    if (err != hipSuccess) {
        fprintf(stderr, "hipSetDevice failed: %s\n", hipGetErrorString(err));
        return EXIT_FAILURE;
    }
    printf("Selected HIP device %d\n", dev);

    // Initialize device memory for this batch of zones
    if (device_init(zones) == EXIT_FAILURE) {
        fprintf(stderr, "Device initialization failed!\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int gpu_backend_finalize(void)
{
    return hip_backend_finalize();
}

void gpu_burner(double* tstep,
                double* temp,
                double* dens,
                double* xin,
                double* xout,
                double* sdotrate,
                unsigned char* burned_zone,
                int* zones)
{
    hyperion_burner_(tstep, temp, dens, xin, xout,
                     sdotrate, burned_zone, zones);
}

#else
#error "ENABLE_GPU defined but USE_HIP is not set"
#endif  /* USE_HIP */

#else  /* ================= CPU FALLBACK ================= */

#include "burner_cpu.h"

int gpu_backend_init(int zones)
{
    (void)zones;
    fprintf(stderr, "GPU backend disabled, using CPU\n");
    fflush(stderr);
    return 0;
}

int gpu_backend_finalize(void)
{
    return 0;
}

void gpu_burner(double* tstep,
                double* temp,
                double* dens,
                double* xin,
                double* xout,
                double* sdotrate,
                unsigned char* burned_zone,
                int* zones)
{
    hyperion_burner_cpu(tstep, temp, dens, xin, xout,
                        sdotrate, burned_zone, zones);
}

#endif /* ENABLE_GPU */


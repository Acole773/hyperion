#include "gpu_backend.h"

#if defined(ENABLE_GPU)

/* select backend */
#if defined(USE_HIP)

#include "hipcore/hip.h"
#include "hipcore/bn_burner_gpu.h"

int device_init(int zones);
void hyperion_burner_(double* tstep,
                      double* temp,
                      double* dens,
                      double* xin,
                      double* xout,
                      double* sdotrate,
                      unsigned char* burned_zone,
                      int* size);

int gpu_backend_init(int zones)
{
    return hip_backend_init(zones);
}

int gpu_backend_finalize(void)
{
#if defined(USE_HIP)
    return hip_backend_finalize();
#elif defined(USE_CUDA)
    return cuda_backend_finalize();
#else
    return 0;
#endif
}

void gpu_burner(
    double* tstep,
    double* temp,
    double* dens,
    double* xin,
    double* xout,
    double* sdotrate,
    unsigned char*  burned_zone,
    int*    zones
) {
    hyperion_burner_(
        tstep, temp, dens, xin, xout,
        sdotrate, burned_zone, zones
    );
}

#elif defined(USE_CUDA)

/* Placeholder for CUDA */
#error "CUDA backend not implemented yet"

#else
#error "ENABLE_GPU set but no backend selected"
#endif

#else  /* CPU fallback */

#include "burner_cpu.h"

int gpu_backend_init(int zones) {
    (void)zones;
    return 0;
}

void gpu_backend_finalize(void) {}

void gpu_burner(
    double* tstep,
    double* temp,
    double* dens,
    double* xin,
    double* xout,
    double* sdotrate,
    unsigned char*  burned_zone,
    int*    size
) {
    hyperion_burner_cpu(
        tstep, temp, dens, xin, xout,
        sdotrate, burned_zone, size
    );
}

#endif


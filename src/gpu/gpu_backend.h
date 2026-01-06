#pragma once

#ifndef GPU_BACKEND_H
#define GPU_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize GPU backend (device selection, allocations, etc.) */
int gpu_backend_init(int zones);

/* Launch burner on GPU */
void gpu_burner(
    double* tstep,
    double* temp,
    double* dens,
    double* xin,
    double* xout,
    double* sdotrate,
    unsigned char* burned_zone,
    int*    size
);

/* Cleanup GPU memory */
int gpu_backend_finalize(void);

#ifdef __cplusplus
}
#endif

#endif /* GPU_BACKEND_H */


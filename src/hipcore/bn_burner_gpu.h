#include <hip/hip_runtime.h>
#include "../core/restrict.h"

#ifndef __BURNER_GPU_H
#define __BURNER_GPU_H
typedef struct {
    double* temp;
    double* dens;
    double* xin;
    double* xout;
    double* sdotrate;

    int* burned_zone;

    double* prefactor;
    double* p_0;
    double* p_1;
    double* p_2;
    double* p_3;
    double* p_4;
    double* p_5;
    double* p_6;

    double* aa;
    double* q_value;

    int* reactant_1;
    int* reactant_2;
    int* reactant_3;

    int* f_plus_map;
    int* f_minus_map;
    double* f_plus_factor;
    double* f_minus_factor;

    int* f_plus_max;
    int* f_minus_max;

    int* num_react_species;
    int* int_vals;
    double* real_vals;
    // exp35: precomputed species processing order (descending by max(p_count, m_count))
    // Used by the kernel to schedule "big" species under wave-cooperative reduction
    // (Phase 1) and "small" species under sub-wave packing (Phase 2). Length SIZE.
    int* species_perm;
} burner_args_t;

#endif

// Kernel Scalar Reals
enum KSR {
    KSR_TSTEP = 0,
};
// Kernel Scalar Ints
// enum KSI {
// };

typedef unsigned char uchar; // Boolean workaround for cython

#ifdef __cplusplus
extern "C" {
#endif

int device_init(int zones);
void hyperion_burner_(double* tstep, double* temp, double* dens, double* xin,
                      double* xout, double* sdotrate, uchar* burned_zone,
                      int* size);
void hip_killall_device_ptrs(void);

#ifdef __cplusplus
}
#endif

// exp34: __restrict__ + const-correctness on kernel pointer params so
// the compiler can reorder LDS/HBM ops without spurious aliasing
// pessimization. Only `xout`, `sdotrate`, and `rate_g` are written by
// the kernel; everything else is read-only after the rate-eval phase.
#ifdef __cplusplus
extern "C" __global__ void hyperion_burner_dev_kernel(
    int zones,
    const double* __restrict__ temp,
    const double* __restrict__ dens,
    const double* __restrict__ xin,
    double*       __restrict__ xout,
    double*       __restrict__ sdotrate,
    const double* __restrict__ prefactor,
    const double* __restrict__ p_0,
    const double* __restrict__ p_1,
    const double* __restrict__ p_2,
    const double* __restrict__ p_3,
    const double* __restrict__ p_4,
    const double* __restrict__ p_5,
    const double* __restrict__ p_6,
    const double* __restrict__ aa,
    const double* __restrict__ q_value,
    const int*    __restrict__ reactant_1,
    const int*    __restrict__ reactant_2,
    const int*    __restrict__ reactant_3,
    const int*    __restrict__ f_plus_map,
    const int*    __restrict__ f_minus_map,
    const double* __restrict__ f_plus_factor,
    const double* __restrict__ f_minus_factor,
    const int*    __restrict__ f_plus_max,
    const int*    __restrict__ f_minus_max,
    const int*    __restrict__ num_react_species,
    const double* __restrict__ real_vals,
    double*       __restrict__ rate_g,
    const int*    __restrict__ species_perm,
    int                          phase1_count);
#endif

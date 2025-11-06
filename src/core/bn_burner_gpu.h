#ifndef __BURNER_GPU_H
#define __BURNER_GPU_H

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

enum KERNEL_ARGS {
    TEMP = 0,
    DENS,
    XIN,
    XOUT,
    SDOTRATE,
    BURNED_ZONE,
    PREFACTOR,
    P_0,
    P_1,
    P_2,
    P_3,
    P_4,
    P_5,
    P_6,
    AA,
    Q_VALUE,
    REACTANT_1,
    REACTANT_2,
    REACTANT_3,
    F_PLUS_MAP,
    F_MINUS_MAP,
    F_PLUS_FACTOR,
    F_MINUS_FACTOR,
    F_PLUS_MAX,
    F_MINUS_MAX,
    NUM_REACT_SPECIES,
    INT_VALS,
    REAL_VALS,
};

// Kernel Scalar Reals
enum KSR {
    KSR_TSTEP = 0,
};
// Kernel Scalar Ints
// enum KSI {
// };

typedef unsigned char uchar; // Boolean workaround for cython

int device_init();
void _killall_ptrs_hipdev();

void hyperion_burner_(double* tstep, double* temp, double* dens, double* xin,
                      double* xout, double* sdotrate, uchar* burned_zone,
                      int* size);

#ifdef __cplusplus
extern "C" __global__ void hyperion_burner_dev_kernel(
    double* temp, double* dens, double* xin, double* xout, double* sdotrate,
    unsigned char* burned_zone, double* prefactor, double* p_0, double* p_1,
    double* p_2, double* p_3, double* p_4, double* p_5, double* p_6, double* aa,
    double* q_value, int* reactant_1, int* reactant_2, int* reactant_3,
    int* f_plus_map, int* f_minus_map, double* f_plus_factor,
    double* f_minus_factor, int* f_plus_max, int* f_minus_max,
    int* num_react_species, int* int_vals, double* real_vals);
#else
__global__ void hyperion_burner_dev_kernel(
    double* temp, double* dens, double* xin, double* xout, double* sdotrate,
    unsigned char* burned_zone, double* prefactor, double* p_0, double* p_1,
    double* p_2, double* p_3, double* p_4, double* p_5, double* p_6, double* aa,
    double* q_value, int* reactant_1, int* reactant_2, int* reactant_3,
    int* f_plus_map, int* f_minus_map, int* f_plus_factor, int* f_minus_factor,
    int* f_plus_max, int* f_minus_max, int* num_react_species, int* int_vals,
    double* real_vals);
#endif

#endif

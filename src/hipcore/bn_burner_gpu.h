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

void hip_killall_device_ptrs(void);

#ifdef __cplusplus
extern "C" {
#endif

int device_init(int zones);
void hyperion_burner_(double* tstep, double* temp, double* dens, double* xin,
                      double* xout, double* sdotrate, uchar* burned_zone,
                      int* size);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" __global__ void hyperion_burner_dev_kernel(
    double* temp, double* dens, double* xin, double* xout, double* sdotrate,
    double* prefactor, double* p_0, double* p_1,
    double* p_2, double* p_3, double* p_4, double* p_5, double* p_6, double* aa,
    double* q_value, int* reactant_1, int* reactant_2, int* reactant_3,
    int* f_plus_map, int* f_minus_map, double* f_plus_factor,
    double* f_minus_factor, int* f_plus_max, int* f_minus_max,
    int* num_react_species, double* real_vals);
#else
__global__ void hyperion_burner_dev_kernel(
    double* temp, double* dens, double* xin, double* xout, double* sdotrate,
    double* prefactor, double* p_0, double* p_1,
    double* p_2, double* p_3, double* p_4, double* p_5, double* p_6, double* aa,
    double* q_value, int* reactant_1, int* reactant_2, int* reactant_3,
    int* f_plus_map, int* f_minus_map, int* f_plus_factor, int* f_minus_factor,
    int* f_plus_max, int* f_minus_max, int* num_react_species, double* real_vals);
#endif

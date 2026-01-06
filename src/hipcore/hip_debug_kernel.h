#ifndef __HIP_DEBUG_KERNEL_H
#define __HIP_DEBUG_KERNEL_H

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" __global__ void vec_mul(double* a, double* b, double* sum);
#else
__global__ void vec_mul(double* a, double* b, double* sum);
#endif

#endif

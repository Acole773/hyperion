#include "hip_debug_kernel.h"

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

extern "C" __global__ void vec_mul(double* a, double* b, double* sum) {
    int i = threadIdx.x;
    sum[i] = a[i] * b[i];
    return;
}

#ifndef __HIP_GPU_H
#define __HIP_GPU_H

#include <hip/hip_runtime.h>

int hip_backend_init(int zones);
int hip_backend_finalize(void);

// Return pointer to hipDeviceProp_t for best device on system. Prints info to
// stdout. Return NULL if error occurs on device creation.
struct hipDeviceProp_t* get_hip_device();

// Test device, printing info to stdout. Return 0 if device fails during test.
int test_device(struct hipDeviceProp_t* device);

#endif

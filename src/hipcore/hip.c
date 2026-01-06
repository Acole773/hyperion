#include "hip.h"
#include "hip_debug_kernel.h"
#include <hip/hip_runtime.h>
#include <stdio.h>

int hip_backend_init(int zones)
{
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        fprintf(stderr, "No HIP devices found\n");
        return EXIT_FAILURE;
    }

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);

    printf("Using HIP device: %s\n", prop.name);
    return EXIT_SUCCESS;
}

int hip_backend_finalize(void)
{
    // optional cleanup
}

struct hipDeviceProp_t* get_hip_device() {
    int num_devices = 0;
    if (hipGetDeviceCount(&num_devices)) {
        printf("ERROR: getting number of devices\n");
        return NULL;
    }

    printf("devices on system: %i\n", num_devices);

    int best_device_id = 0;
    double best_device_tflops = 0;
    for (int i = 0; i < num_devices; i++) {
        struct hipDeviceProp_t temp_device;
        if (hipGetDevicePropertiesR0600(&temp_device, i)) {
            printf("ERROR: cannot get device properties\n");
            return NULL;
        }
        // This is a _really_ loose estimate
        double tflops = ((long)temp_device.clockRate * 1000) *
                        temp_device.multiProcessorCount *
                        temp_device.maxBlocksPerMultiProcessor;
        tflops *= 64;    // Estimated number of operations per cycle (ROPs)
        tflops *= 1e-12; // Convert to tera
        printf("device: %i (%f TFlops/s)\n", i, tflops);
        if (tflops > best_device_tflops) {
            best_device_tflops = tflops;
            best_device_id = i;
        }
    }

    struct hipDeviceProp_t* device =
        (struct hipDeviceProp_t*)malloc(sizeof(struct hipDeviceProp_t));

    if (hipGetDevicePropertiesR0600(device, best_device_id)) {
        printf("ERROR: cannot get device properties\n");
        return NULL;
    }

    printf("using device: %s\n", device->name);
    printf("arch: %s\n", device->gcnArchName);
    printf("dedicated gpu: %s\n", device->integrated ? "no" : "yes");
    printf("clock: %i\n", device->clockRate);
    printf("compute units: %i\n",
           device->multiProcessorCount * device->maxBlocksPerMultiProcessor);
    printf("estimated TFlops/s: %f\n", best_device_tflops);

    hipSetDevice(best_device_id);

    return device;
}

int test_device(struct hipDeviceProp_t* device) {
    int error;

    printf("beginning benchmark (%s)\n", device->name);

    double* h_A = malloc(3 * sizeof(double));
    h_A[0] = 2.0;
    h_A[1] = 4.0;
    h_A[2] = 6.0;
    double* h_B = malloc(3 * sizeof(double));
    h_B[0] = 0.0;
    h_B[1] = 1.0;
    h_B[2] = 2.0;
    double* h_C = malloc(3 * sizeof(double));

    void** arg = malloc(sizeof(void*) * 3);
    hipMalloc(&arg[0], sizeof(double) * 3);
    hipMalloc(&arg[1], sizeof(double) * 3);
    if ((error = hipMalloc(&arg[2], sizeof(double) * 3)) != hipSuccess) {
        printf(
            "ERROR: encountered error allocating device mem: %i\n",
            error);
        return EXIT_FAILURE;
    }

    hipMemcpy(arg[0], h_A, 3 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(arg[1], h_B, 3 * sizeof(double), hipMemcpyHostToDevice);
    if ((error = hipMemcpy(arg[2], h_C, 3 * sizeof(double),
                           hipMemcpyHostToDevice)) != hipSuccess) {
        printf("ERROR: encountered error hip memcpy: %i\n", error);
        return EXIT_FAILURE;
    }

    int sharedmem_allocation = 0 * sizeof(double);
    
    struct dim3 blockdim = {3, 1, 1};
    struct dim3 griddim = {1, 1, 1};
    if ((error = hipConfigureCall(griddim, blockdim, sharedmem_allocation,
                                  hipStreamDefault)) != hipSuccess) {
        printf("ERROR: encountered error configuring kernel: %i\n",
               error);
        return EXIT_FAILURE;
    }

    void** truearg = malloc(sizeof(void*) * 3);
    truearg[0] = &arg[0];
    truearg[1] = &arg[1];
    truearg[2] = &arg[2];
    
    if ((error = hipLaunchKernel(vec_mul, griddim, blockdim, truearg,
                                 sharedmem_allocation, hipStreamDefault)) !=
        hipSuccess) {
        printf("ERROR: encountered error launching kernel: %i\n",
               error);
        return 0;
    }

    hipMemcpy(h_C, arg[2], 3 * sizeof(double), hipMemcpyDeviceToHost);

    printf("Output: {  ");
    for (int i = 0; i < 3; i++) {
        printf("%f  ", h_C[i]);
    }
    
    printf("}\n");
    hipFree(arg[0]);
    hipFree(arg[1]);
    hipFree(arg[2]);
    free(arg);

    return EXIT_SUCCESS;
}

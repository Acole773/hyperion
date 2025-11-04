#include <hip/hip_runtime.h>
#include <iostream>

__global__ void hello_gpu() {
    printf("Hello from GPU thread %d, block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    std::cout << "Hello from CPU\n";
    hello_gpu<<<2, 4>>>();
    hipDeviceSynchronize();
    return 0;
}


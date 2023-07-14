
#include "cudatest.cuh"
#include "common.hpp"

__global__ void kernel()
{
    test();
}

int main()
{

    test();

    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
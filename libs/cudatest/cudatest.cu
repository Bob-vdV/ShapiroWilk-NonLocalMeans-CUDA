#include "cudatest.cuh"

#include <cuda.h>
#include <stdio.h>

__host__ __device__ void test()
{
    printf("testing\n");
}
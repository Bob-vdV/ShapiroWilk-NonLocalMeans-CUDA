#ifndef SWILK_SORT_CUH
#define SWILK_SORT_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace SwilkSort
{
    template <typename T>
    __device__ void heapSort(T arr[], int N);
}
#endif
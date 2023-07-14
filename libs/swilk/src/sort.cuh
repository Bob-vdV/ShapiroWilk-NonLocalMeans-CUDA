#ifndef SWILK_SORT_CUH
#define SWWILK_SORT_CUH

// Heap Sort in C
// Source code taken from: https://www.geeksforgeeks.org/heap-sort/
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace SwilkSort
{
    // Main function to do heap sort
    __host__ __device__ void heapSort(double arr[], int N);

    __device__ void bubbleSort(double arr[], int N);

    // Function to sort an array using
    // insertion sort
    __device__ void insertionSort(double arr[], int n);
}
#endif
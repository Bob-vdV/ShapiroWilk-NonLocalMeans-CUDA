#ifndef BITONICSORT_CUH
#define BITONICSORT_CUH

// Based on article from GeeksForGeeks: https://www.geeksforgeeks.org/bitonic-sort/

/* C++ Program for Bitonic Sort. Note that this program
   works only when size of input is a power of 2. */
#include <bits/stdc++.h>

#include <thrust/swap.h>
#include <thrust/fill.h>

/*The parameter dir indicates the sorting direction, ASCENDING
   or DESCENDING; if (a[i] > a[j]) agrees with the direction,
   then a[i] and a[j] are interchanged.*/
__device__ void compAndSwap(double *a, int i, int j, int dir)
{
    if (dir == (a[i] > a[j]))
    {
        thrust::swap(a[i], a[j]);
    }
}

/*It recursively sorts a bitonic sequence in ascending order,
  if dir = 1, and in descending order otherwise (means dir=0).
  The sequence to be sorted starts at index position low,
  the parameter cnt is the number of elements to be sorted.*/
__device__ void bitonicMerge(double *a, int low, int cnt, int dir)
{
    if (cnt > 1)
    {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++)
        {
            compAndSwap(a, i, i + k, dir);
        }
        bitonicMerge(a, low, k, dir);
        bitonicMerge(a, low + k, k, dir);
    }
}

/* This function first produces a bitonic sequence by recursively
    sorting its two halves in opposite sorting orders, and then
    calls bitonicMerge to make them in the same order */
__device__ void bitonicSort_rec(double *a, int low, int cnt, int dir)
{
    if (cnt > 1)
    {
        int k = cnt / 2;

        // sort in ascending order since dir here is 1
        bitonicSort_rec(a, low, k, 1);

        // sort in descending order since dir here is 0
        bitonicSort_rec(a, low + k, k, 0);

        // Will merge whole sequence in ascending order
        // since dir=1.
        bitonicMerge(a, low, cnt, dir);
    }
}

/**
 * Bitonic sort
 * Prerequisites: a must be allocated to power of 2.
 *
 */
__device__ void bitonicSort(double *a, int size)
{
    const int nearestPow2 = pow(2, ceil(log2((double)size)));

    for (int i = size; i < nearestPow2; i++)
    {
        a[i] = INFINITY;
    }
    // thrust::fill(a + size, a + nearestPow2, INFINITY);
    bitonicSort_rec(a, 0, nearestPow2, 1);
}

//Source: https://stackoverflow.com/questions/73147204/can-bitonic-sort-handle-non-power-of-2-data-in-a-non-recursive-implementation#comment129568489_73158391
template <typename T>
__device__ void comparator(T *a, size_t size, std::size_t i, std::size_t j)
{
    if (i < j && j < size && a[j] < a[i])
        thrust::swap(a[i], a[j]);
}

template <typename T>
__device__ void impBitonicSort(T *a, size_t size)
{
    // Iterate k as if the array size were rounded up to the nearest power of two.
    for (std::size_t k = 2; (k >> 1) < size; k <<= 1)
    {
        for (std::size_t i = 0; i < size; i++)
            comparator(a, size, i, i ^ (k - 1));
        for (std::size_t j = k >> 1; 0 < j; j >>= 1)
            for (std::size_t i = 0; i < size; i++)
                comparator(a, size, i, i ^ j);
    }
}

#endif
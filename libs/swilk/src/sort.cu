// Heap Sort in C
// Source code taken from: https://www.geeksforgeeks.org/heap-sort/
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "sort.cuh"

using namespace SwilkSort;

// Function to swap the position of two elements
template <typename T>
__host__ __device__ void swap2(T &a, T &b)
{
    T temp = a;
    a = b;
    b = temp;
}

// To heapify a subtree rooted with node i
// which is an index in arr[].
// n is size of heap
template <typename T>
__host__ __device__ void heapify(T arr[], int N, int i)
{
    // Find largest among root,
    // left child and right child

    // Initialize largest as root
    int largest = i;

    int left = 2 * i + 1;

    int right = 2 * i + 2;

    // If left child is larger than root
    if (left < N && arr[left] > arr[largest])

        largest = left;

    // If right child is larger than largest
    // so far
    if (right < N && arr[right] > arr[largest])

        largest = right;

    // Swap and continue heapifying
    // if root is not largest
    if (largest != i)
    {

        swap2(arr[i], arr[largest]);

        // Recursively heapify the affected
        // sub-tree
        heapify(arr, N, largest);
    }
}

template __device__ void SwilkSort::heapSort(float arr[], int N);
template __device__ void SwilkSort::heapSort(double arr[], int N);

// Main function to do heap sort
template <typename T>
__device__ void SwilkSort::heapSort(T arr[], int N)
{

    // Build max heap
    for (int i = N / 2 - 1; i >= 0; i--)

        heapify(arr, N, i);

    // Heap sort
    for (int i = N - 1; i >= 0; i--)
    {
        swap2(arr[0], arr[i]);

        // Heapify root element
        // to get highest element at
        // root again
        heapify(arr, i, 0);
    }
}

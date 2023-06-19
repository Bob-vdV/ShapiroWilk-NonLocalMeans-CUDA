#ifndef SWNLM_SORT_CUH
#define SWNLM_SORT_CUH

// Heap Sort in C
// Source code taken from: https://www.geeksforgeeks.org/heap-sort/

namespace
{
    // Function to swap the position of two elements

    __device__ void swap(double *a, double *b)
    {

        double temp = *a;
        *a = *b;
        *b = temp;
    }

    // To heapify a subtree rooted with node i
    // which is an index in arr[].
    // n is size of heap
    __device__ void heapify(double arr[], int N, int i)
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

            swap(&arr[i], &arr[largest]);

            // Recursively heapify the affected
            // sub-tree
            heapify(arr, N, largest);
        }
    }

    // Main function to do heap sort
    __device__ void heapSort(double arr[], int N)
    {

        // Build max heap
        for (int i = N / 2 - 1; i >= 0; i--)

            heapify(arr, N, i);

        // Heap sort
        for (int i = N - 1; i >= 0; i--)
        {
            swap(&arr[0], &arr[i]);

            // Heapify root element
            // to get highest element at
            // root again
            heapify(arr, i, 0);
        }
    }
}
#endif
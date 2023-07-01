#include "swnlmcuda.cuh"
#include "swilk.cuh"

#include <opencv2/core.hpp>

#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_profiler_api.h>

using namespace std;
using namespace cv;

template void swnlmcuda<short, 10, 3>(const Mat &noisyImage, Mat &denoised, const short sigma);
template void swnlmcuda<float, 10, 3>(const Mat &noisyImage, Mat &denoised, const float sigma);
template void swnlmcuda<double, 10, 3>(const Mat &noisyImage, Mat &denoised, const double sigma);
// Based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <typename T>
__device__ void reduceSum(T *array, T val)
{
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    array[idx] = val;
    __syncthreads();

    int i = 2;
    while (i / 2 < blockDim.x * blockDim.y)
    {
        idx = threadIdx.y * blockDim.x + threadIdx.x;
        if (idx % i == 0 && idx + i / 2 < blockDim.x * blockDim.y)
        {
            array[idx] += array[idx + i / 2];
        }

        i *= 2;
        __syncthreads();
    }
}

template <typename T, int searchRadius, int neighborRadius>
__global__ void kernel(const T *in, const double *a, T *out, int rows, int cols, double sigma)
{
    const int padding = searchRadius + neighborRadius;

    // Hacky workaround for using templated extern shared memory
    extern __shared__ char smem[];
    T *searchWindow = reinterpret_cast<T *>(smem);

    // Copy Search window into shared memory
    const size_t searchWindowWidth = (2 * padding + 1);
    const size_t inCols = cols + 2 * padding;

    int i = 0;
    int idx = (threadIdx.y + blockDim.y * i) * searchWindowWidth + threadIdx.x + blockDim.x * i;

    while (idx < searchWindowWidth * searchWindowWidth)
    {
        searchWindow[idx] = in[(blockIdx.y + threadIdx.y + blockDim.y * i) * inCols + blockIdx.x + threadIdx.x + blockDim.x * i];

        i++;
        idx = (threadIdx.y + blockDim.y * i) * searchWindowWidth + threadIdx.x + blockDim.x * i;
    }

    __shared__ double sumWeights;
    __shared__ double avg;

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        sumWeights = 0;
        avg = 0;
    }

    __syncthreads();

    double w = 0;
    double res = 0;

    if (threadIdx.x == searchRadius && threadIdx.y == searchRadius)
    { // center pixel is skipped
        w = 1;
        res = 1 * searchWindow[searchWindowWidth * padding + padding];

        atomicAdd_block(&sumWeights, w);
        atomicAdd_block(&avg, res);
    }
    else
    {
        const int numNeighbors = (neighborRadius * 2 + 1) * (neighborRadius * 2 + 1);

        double *diff = NULL;
        while (diff == NULL)
        {
            diff = new double[numNeighbors];
        }

        const int neighborDiam = neighborRadius * 2 + 1;
        for (int y = 0; y < neighborDiam; y++)
        {
            for (int x = 0; x < neighborDiam; x++)
            {
                const int diffIdx = y * neighborDiam + x;
                const int iNghbrIdx = searchWindowWidth * (y + searchRadius) + x + searchRadius;
                const int jNghbrIdx = searchWindowWidth * (y + threadIdx.y) + x + threadIdx.x;

                diff[diffIdx] = (searchWindow[iNghbrIdx] - searchWindow[jNghbrIdx]) / (sqrt(2.0) * sigma);
            }
        }

        bool hypothesis;
        ShapiroWilk::test(diff, &a[0], numNeighbors, w, hypothesis);

        double mean = 0;
        for (int i = 0; i < numNeighbors; i++)
        {
            mean += diff[i];
        }
        mean /= numNeighbors;

        double stddev = 0;
        for (int i = 0; i < numNeighbors; i++)
        {
            stddev += (diff[i] - mean) * (diff[i] - mean);
        }
        stddev /= numNeighbors;
        stddev = sqrt(stddev);

        delete[] diff;

        double stderror = stddev / neighborDiam; // Neighborhoods are square, thus sqrt(n) observations is number of rows

        if (stderror > mean && mean > -stderror &&
            (1 + stderror > stddev && stddev > 1 - stderror) &&
            hypothesis) // Fail to reject Null hypothesis that it is normally distributed
        {
            res = w * searchWindow[(threadIdx.y + neighborRadius) * searchWindowWidth + threadIdx.x + neighborRadius];

            atomicAdd_block(&sumWeights, w);
            atomicAdd_block(&avg, res);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        const int denoisedIdx = blockIdx.y * cols + blockIdx.x;

        out[denoisedIdx] = min(max(avg / sumWeights, 0.0), 255.0);
    }

    /*
    // W
    __syncthreads();
    reduceSum<T>(searchWindow, w);

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        sumWeights = searchWindow[0];
    }

    // Avg
    __syncthreads();
    reduceSum<T>(searchWindow, res);

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        avg = searchWindow[0];

        const int denoisedIdx = blockIdx.y * cols + blockIdx.x;
        out[denoisedIdx] = avg / sumWeights;
    }*/
}

template <typename T, int searchRadius, int neighborRadius>
void swnlmcuda(const Mat &noisyImage, Mat &denoised, const T sigma)
{
    assert(noisyImage.type() == cv::DataType<T>::type);
    assert(noisyImage.dims == 2);

    const int rows = noisyImage.rows;
    const int cols = noisyImage.cols;

    // Pad the edges with a reflection of the outer pixels.
    const int padding = searchRadius + neighborRadius;
    Mat paddedImage;
    copyMakeBorder(noisyImage, paddedImage, padding, padding, padding, padding, BORDER_REFLECT);

    const int paddedFlat[] = {(int)paddedImage.total()};
    paddedImage = paddedImage.reshape(0, 1, paddedFlat);
    T *h_in = (T *)paddedImage.data;

    const int numNeighbors = (neighborRadius * 2 + 1) * (neighborRadius * 2 + 1);
    vector<double> h_a(numNeighbors + 1);
    ShapiroWilk::setup(h_a.data(), numNeighbors);

    T *d_in, *d_out;
    double *d_a;
    const int inSize = paddedImage.total() * paddedImage.channels() * paddedImage.elemSize();
    cudaMalloc(&d_in, inSize);
    assert(d_in != NULL);
    cudaMemcpyAsync(d_in, h_in, inSize, cudaMemcpyHostToDevice);

    cudaMalloc(&d_a, numNeighbors * sizeof(double));
    assert(d_a != NULL);
    cudaMemcpyAsync(d_a, h_a.data(), numNeighbors * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate output array
    const int flatShape[] = {rows * cols};
    denoised.create(1, flatShape, noisyImage.type());
    T *h_out = (T *)denoised.data;

    const size_t outSize = denoised.total() * denoised.channels() * denoised.elemSize();
    cudaMalloc(&d_out, outSize);
    assert(d_out != NULL);

    const dim3 blocks(rows, cols);

    const int searchDiam = 2 * searchRadius + 1;
    const dim3 threads(searchDiam, searchDiam);

    cudaDeviceSynchronize();

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, (size_t)2 * 1024 * 1024 * 1024); // Set to 2 GB
    const size_t sharedMemSize = (2 * padding + 1) * (2 * padding + 1) * sizeof(T);
    kernel<T, searchRadius, neighborRadius><<<blocks, threads, sharedMemSize>>>(d_in, d_a, d_out, rows, cols, sigma);

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, outSize, cudaMemcpyDeviceToHost);

    const int shape[] = {rows, cols};
    denoised = denoised.reshape(0, 2, shape);

    cudaFree(d_in);
    cudaFree(d_out);

    cudaProfilerStop();
}
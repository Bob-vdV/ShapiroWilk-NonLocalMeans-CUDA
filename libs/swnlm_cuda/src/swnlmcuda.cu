#include "swnlmcuda.cuh"
#include "swilk.cuh"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_profiler_api.h>

using namespace std;
using namespace cv;

template <typename T>
__global__ void kernel(const T *in, const double *a, double *sumWeights, double *avg, int rows, int cols, int searchRadius, int neighborRadius, double sigma)
{
    const size_t searchDiam = 2 * searchRadius + 1;

    const size_t threadNum = blockIdx.x * blockDim.x + threadIdx.x;

    const size_t row = threadNum / (cols *searchDiam * searchDiam); 
    const size_t col = (threadNum / (searchDiam * searchDiam)) % cols;

    const size_t sRow = threadNum % (searchDiam * searchDiam) / searchDiam; // 0 <= sRow < 21 for searchRadius =10
    const size_t sCol = threadNum % searchDiam;

    const size_t padding = searchRadius + neighborRadius;

    const size_t paddedRow = row + padding;
    const size_t paddedCol = col + padding;
    const size_t paddedSRow = paddedRow + sRow - searchRadius;
    const size_t paddedSCol = paddedCol + sCol - searchRadius;

    const size_t inCols = cols + 2 * padding;

    extern __shared__ double diffArr[]; // Shared memory space that is sliced and used for each thread locally

    bool accepted = false;
    double w = 0;
    double res = 0;

    if (paddedSRow == paddedRow && paddedSCol == paddedCol)
    { // center pixel is skipped
        w = 1;
        res = 1 * in[paddedRow * inCols + paddedCol];
        accepted = true;
    }
    else
    {
        const int numNeighbors = (neighborRadius * 2 + 1) * (neighborRadius * 2 + 1);

        double *diff = diffArr + numNeighbors * (threadIdx.x);

        const int neighborDiam = neighborRadius * 2 + 1;
        for (int y = 0; y < neighborDiam; y++)
        {
            for (int x = 0; x < neighborDiam; x++)
            {
                const int diffIdx = y * neighborDiam + x;

                const int iNghbrIdx = inCols * (paddedRow + y - neighborRadius) + paddedCol + x - neighborRadius;
                const int jNghbrIdx = inCols * (paddedSRow + y - neighborRadius) + paddedSCol + x - neighborRadius;

                diff[diffIdx] = (in[iNghbrIdx] - in[jNghbrIdx]) / (sqrt(2.0) * sigma);
            }
        }

        bool hypothesis;
        ShapiroWilk::test(diff, a, numNeighbors, w, hypothesis);

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

        const double stderror = stddev / neighborDiam; // Neighborhoods are square, thus sqrt(n) observations is number of rows

        if (stderror > mean && mean > -stderror &&
            (1 + stderror > stddev && stddev > 1 - stderror) &&
            hypothesis) // Fail to reject Null hypothesis that it is normally distributed
        {
            res = w * in[paddedSRow * inCols + paddedSCol];
            accepted = true;
        }
    }
    if (accepted)
    {
        atomicAdd(&sumWeights[row * cols + col], w);
        atomicAdd(&avg[row * cols + col], res);
    }
}

//
template <typename T>
__global__ void denoiseStep(double *sumWeights, double *avg, T *out, const int rows, const int cols)
{
    const int threadNum = blockIdx.x * blockDim.x + threadIdx.x;

    out[threadNum] = avg[threadNum] / sumWeights[threadNum];
}

template void swnlmcuda(const Mat &noisyImage, Mat &denoised, const short sigma, const int searchRadius, const int neighborRadius);
template void swnlmcuda(const Mat &noisyImage, Mat &denoised, const float sigma, const int searchRadius, const int neighborRadius);
template void swnlmcuda(const Mat &noisyImage, Mat &denoised, const double sigma, const int searchRadius, const int neighborRadius);

template <typename T>
void swnlmcuda(const Mat &noisyImage, Mat &denoised, const T sigma, const int searchRadius, const int neighborRadius)
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

    const int searchDiam = 2 * searchRadius + 1;

    const int totalThreads = rows * cols * searchDiam * searchDiam;
    const int threadsPerBlock = 32; // Arbitrarily chosen, needs more experimentation.
    const int numBlocks = ceil((double)totalThreads / threadsPerBlock);

    dim3 blocks(numBlocks);
    dim3 threads(threadsPerBlock);

    T *d_in, *d_out;
    double *d_a;
    const size_t inSize = paddedImage.total() * paddedImage.channels() * paddedImage.elemSize();
    cudaMalloc(&d_in, inSize);
    assert(d_in != NULL);
    cudaMemcpyAsync(d_in, h_in, inSize, cudaMemcpyHostToDevice);

    cudaMalloc(&d_a, numNeighbors * sizeof(double));
    assert(d_a != NULL);
    cudaMemcpyAsync(d_a, h_a.data(), numNeighbors * sizeof(double), cudaMemcpyHostToDevice);

    double *d_sumWeights, *d_avg;
    cudaMalloc(&d_sumWeights, rows * cols * sizeof(double));
    assert(d_sumWeights != NULL);
    cudaMemset(d_sumWeights, 0, rows * cols * sizeof(double));

    cudaMalloc(&d_avg, rows * cols * sizeof(double));
    assert(d_avg != NULL);
    cudaMemset(d_avg, 0, rows * cols * sizeof(double));

    // Allocate output array
    const int flatShape[] = {rows * cols};
    denoised.create(1, flatShape, noisyImage.type());
    T *h_out = (T *)denoised.data;

    const size_t sharedMemSize = numNeighbors * threadsPerBlock * sizeof(double);
    cudaDeviceSynchronize();

    kernel<T><<<blocks, threads, sharedMemSize>>>(d_in, d_a, d_sumWeights, d_avg, rows, cols, searchRadius, neighborRadius, sigma);

    const size_t outSize = denoised.total() * denoised.channels() * denoised.elemSize();
    cudaMalloc(&d_out, outSize);
    assert(d_out != NULL);

    dim3 denoiseBlocks(ceil((double)rows * cols / threadsPerBlock));
    dim3 denoiseThreads(threadsPerBlock);

    cudaDeviceSynchronize();

    denoiseStep<T><<<denoiseBlocks, denoiseThreads>>>(d_sumWeights, d_avg, d_out, rows, cols);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, outSize, cudaMemcpyDeviceToHost);

    const int shape[] = {rows, cols};
    denoised = denoised.reshape(0, 2, shape);

    cudaFree(d_in);
    cudaFree(d_a);
    cudaFree(d_sumWeights);
    cudaFree(d_avg);
    cudaFree(d_out);

    cudaProfilerStop();
}
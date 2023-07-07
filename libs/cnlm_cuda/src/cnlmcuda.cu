#include "cnlmcuda.cuh"
#include "common.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_profiler_api.h>

using namespace std;
using namespace cv;

template <typename T>
__global__ void calculateWeights(const T *in, const double *kernel, double *sumWeights, double *avg, int rows, int cols, int searchRadius, int neighborRadius, double sigma)
{
    const double h = 1 * sigma;

    const size_t searchDiam = 2 * searchRadius + 1;

    const size_t threadNum = blockIdx.x * blockDim.x + threadIdx.x;

    const size_t row = threadNum / (rows * searchDiam * searchDiam);
    const size_t col = (threadNum / (searchDiam * searchDiam)) % rows;

    const size_t sRow = threadNum % (searchDiam * searchDiam) / searchDiam; // 0 <= sRow < 21 for searchRadius =10
    const size_t sCol = threadNum % searchDiam;

    const size_t padding = searchRadius + neighborRadius;

    const size_t paddedRow = row + padding;
    const size_t paddedCol = col + padding;
    const size_t paddedSRow = paddedRow + sRow - searchRadius;
    const size_t paddedSCol = paddedCol + sCol - searchRadius;

    const size_t inCols = cols + 2 * padding;

    double sum = 0;

    const int neighborDiam = neighborRadius * 2 + 1;
    for (int y = 0; y < neighborDiam; y++)
    {
        for (int x = 0; x < neighborDiam; x++)
        {
            const int kernelIdx = y * neighborDiam + x;

            const int iNghbrIdx = inCols * (paddedRow + y - neighborRadius) + paddedCol + x - neighborRadius;
            const int jNghbrIdx = inCols * (paddedSRow + y - neighborRadius) + paddedSCol + x - neighborRadius;

            sum += pow(in[iNghbrIdx] - in[jNghbrIdx], 2) * kernel[kernelIdx];
        }
    }
    double weight = exp(-sum / (h * h));
    double res = weight * in[paddedSRow * inCols + paddedSCol];

    atomicAdd(&sumWeights[row * cols + col], weight);
    atomicAdd(&avg[row * cols + col], res);
}

template <typename T>
__global__ void denoiseStep(double *sumWeights, double *avg, T *out, const int rows, const int cols)
{
    const int threadNum = blockIdx.x * blockDim.x + threadIdx.x;

    out[threadNum] = avg[threadNum] / sumWeights[threadNum];
}

template void cnlmcuda(const Mat &noisyImage, Mat &denoised, const short sigma, const int searchRadius, const int neighborRadius);
template void cnlmcuda(const Mat &noisyImage, Mat &denoised, const float sigma, const int searchRadius, const int neighborRadius);
template void cnlmcuda(const Mat &noisyImage, Mat &denoised, const double sigma, const int searchRadius, const int neighborRadius);

template <typename T>
void cnlmcuda(const Mat &noisyImage, Mat &denoised, const T sigma, const int searchRadius, const int neighborRadius)
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

    vector<double> gaussKernel;
    makeGaussianKernel(gaussKernel, neighborRadius);
    double *h_kernel = gaussKernel.data();

    const int numNeighbors = (neighborRadius * 2 + 1) * (neighborRadius * 2 + 1);
    const int searchDiam = 2 * searchRadius + 1;

    const int totalThreads = rows * cols * searchDiam * searchDiam;
    const int threadsPerBlock = 32; // Should be multiple of 32 (warp size). Empirically 32 seems the fastest, though there is little variation.
    const int numBlocks = ceil((double)totalThreads / threadsPerBlock);

    dim3 blocks(numBlocks);
    dim3 threads(threadsPerBlock);

    T *d_in, *d_out;
    double *d_kernel;
    const size_t inSize = paddedImage.total() * paddedImage.channels() * paddedImage.elemSize();
    cudaMalloc(&d_in, inSize);
    assert(d_in != NULL);
    cudaMemcpyAsync(d_in, h_in, inSize, cudaMemcpyHostToDevice);

    cudaMalloc(&d_kernel, numNeighbors * sizeof(double));
    assert(d_kernel != NULL);
    cudaMemcpyAsync(d_kernel, h_kernel, numNeighbors * sizeof(double), cudaMemcpyHostToDevice);

    double *d_sumWeights, *d_avg;
    cudaMalloc(&d_sumWeights, rows * cols * sizeof(double));
    assert(d_sumWeights != NULL);
    cudaMemsetAsync(d_sumWeights, 0, rows * cols * sizeof(double));

    cudaMalloc(&d_avg, rows * cols * sizeof(double));
    assert(d_avg != NULL);
    cudaMemsetAsync(d_avg, 0, rows * cols * sizeof(double));

    // Allocate output array
    const int flatShape[] = {rows * cols};
    denoised.create(1, flatShape, noisyImage.type());
    T *h_out = (T *)denoised.data;

    cudaDeviceSynchronize();
    calculateWeights<T><<<blocks, threads>>>(d_in, d_kernel, d_sumWeights, d_avg, rows, cols, searchRadius, neighborRadius, sigma);

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
    cudaFree(d_sumWeights);
    cudaFree(d_avg);
    cudaFree(d_out);

    cudaProfilerStop();
}
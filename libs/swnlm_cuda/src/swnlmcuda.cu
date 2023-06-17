#include "swnlmcuda.cuh"
#include "swilk.cuh"

#include <opencv2/core.hpp>

#include <iostream>
#include <vector>

#include <cuda.h>

using namespace std;
using namespace cv;

__device__ void atomicMax_block(double *addr, double val){
    unsigned long long int cmp = *addr < val;
    atomicCAS_block((unsigned long long int* )addr, cmp, *(unsigned long long int*)&val);
}

__global__ void kernel(const double *in, const double *a, double *out, int rows, int cols, int searchRadius, int neighborRadius, double sigma, double alpha)
{
    const int padding = searchRadius + neighborRadius;

    const int row = padding + blockIdx.y;
    const int col = padding + blockIdx.x;

    const int sRow = threadIdx.y + row - searchRadius;
    const int sCol = threadIdx.x + col - searchRadius;

    const int numNeighbors = (neighborRadius * 2 + 1) * (neighborRadius * 2 + 1);

    __shared__ double Wmax;
    __shared__ double avg;
    __shared__ double sumWeights;

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        Wmax = 0;
        avg = 0;
        sumWeights = 0;
    }
    __syncthreads();

    if (sRow == row && sCol == col)
    { // center pixel is skipped
        return;
    }

    double *diff = new double[numNeighbors];
    const int neighborDiam = neighborRadius * 2 + 1;
    for (int y = 0; y < neighborDiam; y++)
    {
        for (int x = 0; x < neighborDiam; x++)
        {
            const int diffIdx = y * neighborDiam + x;
            const int iNghbrIdx = (2 * padding + cols) * (row + y - neighborRadius) + col + x - neighborRadius;
            const int jNghbrIdx = (2 * padding + cols) * (sRow + y - neighborRadius) + sCol + x - neighborRadius;

            diff[diffIdx] = (in[iNghbrIdx] - in[jNghbrIdx]) / (sqrt(2.0) * sigma);
        }
    }

    double w, pw;
    ShapiroWilk::test(diff, a, numNeighbors, w, pw);

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
        (pw > alpha)) // Fail to reject Null hypothesis that it is normally distributed
    {

        // ATOMIC OPERATIONS::

        atomicMax_block(&Wmax, w);

        atomicAdd_block(&sumWeights, w);

        atomicAdd_block(&avg, w * in[(2 * padding + cols) * sRow + sCol]);
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        avg += Wmax * in[(2 * padding + cols) * row + col];
        sumWeights += Wmax;

        const int denoisedIdx = (row - padding) * cols + col - padding;

        if (sumWeights > 0)
        {
            out[denoisedIdx] = avg / sumWeights;
        }
        else
        {
            out[denoisedIdx] = in[(2 * padding + cols) * row + col];
        }
    }
}

void swnlmcuda(const Mat &noisyImage, Mat &denoised, const double sigma, const int searchRadius, const int neighborRadius)
{


    assert(noisyImage.type() == CV_64FC1);
    assert(noisyImage.dims == 2);

    const int rows = noisyImage.rows;
    const int cols = noisyImage.cols;

    double alpha = 0.05; // threshold  for accepting null hypothesis. Typically is 0.05 for statistics

    // Pad the edges with a reflection of the outer pixels.
    const int padding = searchRadius + neighborRadius;
    Mat paddedImage;
    copyMakeBorder(noisyImage, paddedImage, padding, padding, padding, padding, BORDER_REFLECT);

    const int paddedFlat[] = {(int)paddedImage.total()};
    paddedImage = paddedImage.reshape(0, 1, paddedFlat);
    double *h_in = (double *)paddedImage.data;

    const int numNeighbors = (neighborRadius * 2 + 1) * (neighborRadius * 2 + 1);
    vector<double> a(numNeighbors + 1);
    ShapiroWilk::setup(a.data(), numNeighbors);

    double *d_in, *d_a, *d_out;
    const int inSize = paddedImage.total() * paddedImage.channels() * paddedImage.elemSize();
    cudaMalloc(&d_in, inSize);
    assert(d_in != NULL);
    cudaMemcpy(d_in, h_in, inSize, cudaMemcpyHostToDevice);

    cudaMalloc(&d_a, (numNeighbors + 1) * sizeof(double));
    assert(d_a != NULL);
    cudaMemcpy(d_a, a.data(), (numNeighbors + 1) * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate output array
    const int flatShape[] = {rows * cols};
    denoised.create(1, flatShape, noisyImage.type());
    double *h_out = (double *)denoised.data;

    const size_t outSize = denoised.total() * denoised.channels() * denoised.elemSize();
    cudaMalloc(&d_out, outSize);
    assert(d_out != NULL);



    const dim3 blocks(rows, cols);

    const int searchDiam = 2 * searchRadius + 1;
    const dim3 threads(searchDiam, searchDiam);

    kernel<<<blocks, threads>>>(d_in, d_a, d_out, rows, cols, searchRadius, neighborRadius, sigma, alpha);

    /*
    for (int row = padding; row < rows + padding; row++)
    {
        for (int col = padding; col < cols + padding; col++)
        {
            Range searchRanges[] = {Range(row - searchRadius, row + searchRadius + 1), Range(col - searchRadius, col + searchRadius + 1)};
            const int numSearches = (2 * searchRadius + 1) * (2 * searchRadius + 1);
            const int searchShape[] = {numSearches};
            Mat searchWindow = paddedImage(searchRanges).clone().reshape(0, 1, searchShape);
            double *h_s = (double *)searchWindow.data;

            // TODO laucnh kernels, then collect sums later

            kernel<<<1, >>>(d_in, const double *a, double *out, int rows, int cols, int searchRadius, int neighborRadius, double sigma, double alpha))

            avg += Wmax * in[(2 * padding + cols) * row + col];
            sumWeights += Wmax;

            const int denoisedIdx = (row - padding) * cols + col - padding;
            if (sumWeights > 0)
            {
                out[denoisedIdx] = avg / sumWeights;
            }
            else
            {
                out[denoisedIdx] = in[(2 * padding + cols) * row + col];
            }
        }
    }*/
    /*
        const int flatShape[] = {rows * cols};
        denoised.create(1, flatShape, noisyImage.type());
        double *h_out = (double *)denoised.data;

        const int blockSize = 512;

        double *d_in, *d_out, *d_a;

        // Allocate input arrays
        const int inSize = paddedImage.total() * paddedImage.channels() * paddedImage.elemSize();
        cudaMalloc(&d_in, inSize);
        cudaMemcpy(d_in, h_in, inSize, cudaMemcpyHostToDevice);

        cudaMalloc(&d_a, (numNeighbors + 1) * sizeof(double));
        cudaMemcpy(d_a, a.data(), (numNeighbors + 1) * sizeof(double), cudaMemcpyHostToDevice);

        // Allocate output array
        const int outSize = denoised.total() * denoised.channels() * denoised.elemSize();
        cudaMalloc(&d_out, outSize);

        kernel<<<rows, cols>>>(d_in, d_a, d_out, rows, cols, searchRadius, neighborRadius, sigma, alpha);

        */

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, outSize, cudaMemcpyDeviceToHost);

    const int shape[] = {rows, cols};
    denoised = denoised.reshape(0, 2, shape);

    cudaFree(d_in);
    cudaFree(d_a);
    cudaFree(d_out);
}
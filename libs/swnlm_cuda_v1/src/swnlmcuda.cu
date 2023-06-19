#include "swnlmcuda.cuh"
#include "swilk.cuh"
#include "helper_cuda.h"

#include <opencv2/core.hpp>

#include <iostream> //TODO remove

using namespace std;
using namespace cv;

#define NEIGHBORSIZE 7

__device__ inline double getElem(const double *array, const int rows, const int cols, const int row, const int col)
{
    return array[cols * row + col];
}

__global__ void applySWNLM(const double *noisyImage, double *denoised, const double *coefficients, const int rows, const int cols, const double sigma, const int searchRadius, const int neighborRadius)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    const int padding = searchRadius + neighborRadius;

    double Wmax = 0;
    double avg = 0;
    double sumWeights = 0;

    double neighborDiff[NEIGHBORSIZE * NEIGHBORSIZE];

    // const Range rangesI[] = {Range(row - neighborRadius, row + neighborRadius + 1), Range(col - neighborRadius, col + neighborRadius + 1)};
    // const Mat neighborhoodI = paddedImage(rangesI);

    // double neighborhoodDiff[neighborRadius * neighborRadius];

    const double alpha = 0.05;

    for (int sRow = row - searchRadius; sRow <= row + searchRadius; sRow++)
    {
        for (int sCol = col - searchRadius; sCol <= col + searchRadius; sCol++)
        {
            if (sRow == row && sCol == col)
            { // center pixel is skipped
                continue;
            }

            // const Range rangesJ[] = {Range(sRow - neighborRadius, sRow + neighborRadius + 1), Range(sCol - neighborRadius, sCol + neighborRadius + 1)};
            // const Mat neighborhoodJ = paddedImage(rangesJ);

            for (int nRow = 0; nRow < NEIGHBORSIZE; nRow++)
            {
                for (int nCol = 0; nCol < NEIGHBORSIZE; nCol++)
                {
                    const int offsetRow = nRow - neighborRadius;
                    const int offsetCol = nCol - neighborRadius;
                    const double valI = getElem(noisyImage, rows, cols, row + offsetRow, col + offsetCol);
                    const double valJ = getElem(noisyImage, rows, cols, sRow + offsetRow, sCol + offsetCol);

                    neighborDiff[nRow * NEIGHBORSIZE + nCol] = (valI - valJ) / (sqrt(2.0) * sigma);
                }
            }

            // Mat difference = neighborhoodI - neighborhoodJ;
            // Mat normalized = difference / (sqrt(2) * sigma);

            // assert(normalized.isContinuous());
            // std::vector<double> array;
            // array.assign((double *)normalized.data, (double *)normalized.data + normalized.total());

            double w, pw;
            swilk::test(neighborDiff, coefficients, NEIGHBORSIZE * NEIGHBORSIZE, w, pw);

            // cout << w << '\t' << pw << '\n';
            double mean = 0;
            for (int i = 0; i < NEIGHBORSIZE * NEIGHBORSIZE; i++)
            {
                mean += neighborDiff[i];
            }
            mean /= NEIGHBORSIZE * NEIGHBORSIZE;

            double stdev = 0;
            for (int i = 0; i < NEIGHBORSIZE * NEIGHBORSIZE; i++)
            {
                stdev += (neighborDiff[i] - mean) * (neighborDiff[i] - mean);
            }
            stdev /= (NEIGHBORSIZE * NEIGHBORSIZE);
            stdev = sqrt(stdev);

            double stderror = stdev / NEIGHBORSIZE; // Neighborhoods are square, thus sqrt(n) observations is number of rows

            if (stderror > mean && mean > -stderror &&
                (1 + stderror > stdev && stdev > 1 - stderror) &&
                pw > alpha)
            {
                Wmax = max(w, Wmax);

                sumWeights += w;
                avg += w * getElem(noisyImage, rows, cols, sRow, sCol);
            }
        }
    }
    avg += Wmax * getElem(noisyImage, rows, cols, row, col);
    sumWeights += Wmax;
    const size_t idx = (row - padding) * cols + col - padding;

    if (sumWeights > 0)
    {
        denoised[idx] = avg / sumWeights;
    }
    else
    {
        denoised[idx] = getElem(noisyImage, rows, cols, row, col);
    }
}

void swnlmcuda(const cv::Mat &noisyImage, cv::Mat &denoised, const double sigma, const int searchRadius, const int neighborRadius)
{
    const int rows = noisyImage.rows;
    const int cols = noisyImage.cols;
    const int type = noisyImage.type();

    // Pad the edges with a reflection of the outer pixels.
    const int padding = searchRadius + neighborRadius;
    Mat paddedImage;
    copyMakeBorder(noisyImage, paddedImage, padding, padding, padding, padding, BORDER_REFLECT);

    Mat flattened = paddedImage.reshape(0, paddedImage.total());
    double *input, *output;

    const size_t size = flattened.total() * flattened.elemSize();

    checkCudaErrors(cudaMallocManaged(&input, size));
    checkCudaErrors(cudaMemcpy(input, flattened.data, size, cudaMemcpyDefault));

    checkCudaErrors(cudaMallocManaged(&output, size));

    const size_t neighborHoodSize = NEIGHBORSIZE * NEIGHBORSIZE;
    vector<double> coeff(neighborHoodSize + 1);
    swilk::setup(coeff.data(), neighborHoodSize);

    double *coefficients;
    checkCudaErrors(cudaMallocManaged(&coefficients, (neighborHoodSize + 1) * sizeof(double)));
    checkCudaErrors(cudaMemcpy(coefficients, coeff.data(), (neighborHoodSize + 1) * sizeof(double), cudaMemcpyDefault));

    const dim3 blockDims(16, 16);
    const int blockSize = blockDims.x * blockDims.y * blockDims.z;
    const int numBlocks = (flattened.total() + blockSize - 1) / blockSize;

    applySWNLM<<<1,1>>>/*<<<numBlocks, blockSize>>>*/(input, output, coefficients, rows, cols, sigma, searchRadius, neighborRadius);

    checkCudaErrors(cudaDeviceSynchronize());

    const int flatShape[] = {rows * cols};
    flattened.create(1, flatShape, type);
    checkCudaErrors(cudaMemcpy(flattened.data, output, size, cudaMemcpyDefault));

    checkCudaErrors(cudaFree(input));
    checkCudaErrors(cudaFree(output));
    checkCudaErrors(cudaFree(coefficients));

    const int shape[] = {rows, cols};
    denoised = flattened.reshape(0, 2, shape);

    cout << denoised.size << '\n';
}

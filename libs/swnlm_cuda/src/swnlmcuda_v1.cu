#include "swnlmcuda.cuh"
#include "swilk.cuh"

#include <opencv2/core.hpp>

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

__global__ void kernel(const double *in, const double *a, double *out, int rows, int cols, int searchRadius, int neighborRadius, double sigma, double alpha)
{
    const int padding = searchRadius + neighborRadius;

    //TODO: dynamic work division
    const int row = blockIdx.x + padding;
    const int col = threadIdx.x + padding;

    const int numNeighbors = (neighborRadius * 2 + 1) * (neighborRadius * 2 + 1);

    double Wmax = 0;
    double avg = 0;
    double sumWeights = 0;

    for (int sRow = row - searchRadius; sRow <= row + searchRadius; sRow++)
    {
        for (int sCol = col - searchRadius; sCol <= col + searchRadius; sCol++)
        {
            if (sRow == row && sCol == col)
            { // center pixel is skipped
                continue;
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

            // cout << w << '\t' << pw << '\n';
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
                (pw > alpha)) // Fail to reject Null hypothesis that it is not normally distributed
            {
                Wmax = max(w, Wmax);

                sumWeights += w;

                avg += w * in[(2 * padding + cols) * sRow + sCol];
            }
        }
    }
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

    const int paddedFlat[] = { (int) paddedImage.total()};
    paddedImage = paddedImage.reshape(0, 1, paddedFlat);
    double *h_in = (double *)paddedImage.data;

    const int numNeighbors = (neighborRadius * 2 + 1) * (neighborRadius * 2 + 1);
    vector<double> a(numNeighbors + 1);
    ShapiroWilk::setup(a.data(), numNeighbors);

    const int flatShape[] = {rows * cols};
    denoised.create(1, flatShape, noisyImage.type());
    double *h_out = (double *)denoised.data;

    const int blockSize = 512;

    double *d_in, *d_out, *d_a;

    //Allocate input arrays
    const int inSize = paddedImage.total() * paddedImage.channels() * paddedImage.elemSize();
    cudaMalloc(&d_in, inSize);
    cudaMemcpy(d_in, h_in, inSize, cudaMemcpyHostToDevice);

    cudaMalloc(&d_a, (numNeighbors + 1) * sizeof(double));
    cudaMemcpy(d_a, a.data(), (numNeighbors + 1) * sizeof(double), cudaMemcpyHostToDevice);


    // Allocate output array
    const int outSize = denoised.total() * denoised.channels() * denoised.elemSize();
    cudaMalloc(&d_out, outSize);

    kernel<<<rows, cols>>>(d_in, d_a, d_out, rows, cols, searchRadius, neighborRadius, sigma, alpha);

    cudaMemcpy(h_out, d_out, outSize, cudaMemcpyDeviceToHost);

    const int shape[] = {rows, cols};
    denoised = denoised.reshape(0, 2, shape);

    cudaFree(d_in);
    cudaFree(d_a);
    cudaFree(d_out);
}
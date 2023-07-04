#include "cnlm.hpp"

#include <opencv2/core.hpp>

// todo remove
#include <opencv2/highgui.hpp>

#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

void makeGaussianKernel(vector<double> &gaussKernel, const int neighborRadius)
{
    assert(type == CV_64FC1);

    const int rows = neighborRadius * 2 + 1;
    const int cols = neighborRadius * 2 + 1;

    const int stdev = 1;
    const int middle = neighborRadius;

    gaussKernel.resize(rows * cols);

    double *kernel = gaussKernel.data();

    double sum = 0;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            const int rowDist = row - middle;
            const int colDist = col - middle;

            kernel[row * cols + col] = exp(((rowDist * rowDist) + (colDist * colDist)) / (-2.0 * stdev * stdev));
            sum += kernel[row * cols + col];
        }
    }

    for (int elem = 0; elem < rows * cols; elem++)
    {
        kernel[elem] /= sum;
    }
}

void cnlm(const cv::Mat &noisyImage, cv::Mat &denoised, const double sigma, const int searchRadius, const int neighborRadius)
{
    assert(noisyImage.type() == CV_64FC1);
    assert(noisyImage.dims == 2);

    const int rows = noisyImage.rows;
    const int cols = noisyImage.cols;
    const double h = 1 * sigma;

    // Pad the edges with a reflection of the outer pixels.
    const int padding = searchRadius + neighborRadius;
    Mat paddedImage;
    copyMakeBorder(noisyImage, paddedImage, padding, padding, padding, padding, BORDER_REFLECT);

    const int paddedFlat[] = {(int)paddedImage.total()};
    paddedImage = paddedImage.reshape(0, 1, paddedFlat);
    double *in = (double *)paddedImage.data;

    vector<double> gaussKernel;
    makeGaussianKernel(gaussKernel, neighborRadius);
    double *kernel = gaussKernel.data();

    const int flatShape[] = {rows * cols};
    denoised.create(1, flatShape, noisyImage.type());
    double *denoisedOut = (double *)denoised.data;

    const int inCols = cols + 2 * padding;

    for (int row = padding; row < rows + padding; row++)
    {
        for (int col = padding; col < cols + padding; col++)
        {
            double sumWeights = 0;
            double val = 0;

            for (int sRow = row - searchRadius; sRow <= row + searchRadius; sRow++)
            {
                for (int sCol = col - searchRadius; sCol <= col + searchRadius; sCol++)
                {
                    double sum = 0;

                    const int neighborDiam = neighborRadius * 2 + 1;
                    for (int y = 0; y < neighborDiam; y++)
                    {
                        for (int x = 0; x < neighborDiam; x++)
                        {
                            const int iNghbrIdx = (2 * padding + cols) * (row + y - neighborRadius) + col + x - neighborRadius;
                            const int jNghbrIdx = (2 * padding + cols) * (sRow + y - neighborRadius) + sCol + x - neighborRadius;
                            const int gaussKernelIdx = y * neighborDiam + x;

                            sum += pow(in[iNghbrIdx] - in[jNghbrIdx], 2) * kernel[gaussKernelIdx];
                        }
                    }
                    double weight = exp(-sum / (h * h));
                    sumWeights += weight;
                    val += weight * in[sRow * inCols + sCol];
                }
            }
            val /= sumWeights;
            const int denoisedIdx = (row - padding) * cols + col - padding;
            denoisedOut[denoisedIdx] = val;
        }
        const int shape[] = {rows, cols};
        denoised = denoised.reshape(0, 2, shape);
    }
}
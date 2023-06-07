#include "cnlm.hpp"

#include <opencv2/core.hpp>

// todo remove
#include <opencv2/highgui.hpp>

#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

void makeGaussianKernel(Mat &gaussKernel, const int neighborRadius, const int type)
{
    assert(type == CV_64FC1);

    const int rows = neighborRadius * 2 + 1;
    const int cols = neighborRadius * 2 + 1;

    const int stdev = 1;
    const int middle = neighborRadius;

    const int shape[] = {rows, cols};
    gaussKernel.create(2, shape, type);

    double sum = 0;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            const int rowDist = row - middle;
            const int colDist = col - middle;

            gaussKernel.at<double>(row, col) = exp(((rowDist * rowDist) + (colDist * colDist)) / (-2.0 * stdev * stdev));
            sum += gaussKernel.at<double>(row, col);
        }
    }
    gaussKernel /= sum;
}

void cnlm(const cv::Mat &noisyImage, cv::Mat &denoised, const double sigma, const int searchRadius, const int neighborRadius)
{
    assert(noisyImage.type() == CV_64FC1);
    assert(noisyImage.dims == 2);

    const int rows = noisyImage.rows;
    const int cols = noisyImage.cols;
    const int type = noisyImage.type();
    const double h = 1 * sigma;

    // Pad the edges with a reflection of the outer pixels.
    const int padding = searchRadius + neighborRadius;
    Mat paddedImage;
    copyMakeBorder(noisyImage, paddedImage, padding, padding, padding, padding, BORDER_REFLECT);

    Mat gaussKernel;
    makeGaussianKernel(gaussKernel, neighborRadius, type);

    const int shape[] = {rows, cols};
    denoised.create(2, shape, type);
    for (int row = padding; row < rows + padding; row++)
    {
        for (int col = padding; col < cols + padding; col++)
        {
            const Range rangesI[] = {
                Range(row - neighborRadius, row + neighborRadius + 1),
                Range(col - neighborRadius, col + neighborRadius + 1),
            };
            const Mat neighborsI = paddedImage(rangesI);

            double sumWeights = 0;
            double val = 0;

            for (int sRow = row - searchRadius; sRow <= row + searchRadius; sRow++)
            {
                for (int sCol = col - searchRadius; sCol <= col + searchRadius; sCol++)
                {
                    const Range rangesJ[] = {
                        Range(sRow - neighborRadius, sRow + neighborRadius + 1),
                        Range(sCol - neighborRadius, sCol + neighborRadius + 1)};
                    const Mat neighborsJ = paddedImage(rangesJ);
                    Mat diff = neighborsI - neighborsJ;
                    diff = diff.mul(diff);

                    const double sum = cv::sum(diff.mul(gaussKernel))[0];
                    const double weight = exp(-sum / (h * h));

                    sumWeights += weight;
                    val += weight * paddedImage.at<double>(sRow, sCol);
                }
            }
            val /= sumWeights;
            denoised.at<double>(row - padding, col - padding) = val;
        }
    }
}

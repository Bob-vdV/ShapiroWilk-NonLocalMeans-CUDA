#include "swnlm.hpp"
#include "swilk.hpp"

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

void swnlm(const Mat &noisyImage, Mat &denoised, const double sigma, const int searchRadius, const int neighborRadius)
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

    const int paddedFlat[] = {paddedImage.total()};
    paddedImage = paddedImage.reshape(0, 1, paddedFlat);
    double *in = (double *)paddedImage.data;

    const int numNeighbors = (neighborRadius * 2 + 1) * (neighborRadius * 2 + 1);
    ShapiroWilk swilk(numNeighbors);

    const int flatShape[] = {rows * cols};
    denoised.create(1, flatShape, noisyImage.type());
    double *denoisedOut = (double *)denoised.data;

    for (int row = padding; row < rows + padding; row++)
    {
        for (int col = padding; col < cols + padding; col++)
        {
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
                    swilk.test(diff, w, pw);

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
                denoisedOut[denoisedIdx] = avg / sumWeights;
            }
            else
            {
                denoisedOut[denoisedIdx] = in[(2 * padding + cols) * row + col];
            }
        }
    }

    const int shape[] = {rows, cols};
    denoised = denoised.reshape(0, 2, shape);
}
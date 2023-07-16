#include "swnlm.hpp"
#include "swilk.cuh"

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

template void swnlm(const Mat &noisyImage, Mat &denoised, const uint8_t sigma, const int searchRadius, const int neighborRadius);
template void swnlm(const Mat &noisyImage, Mat &denoised, const int32_t sigma, const int searchRadius, const int neighborRadius);
template void swnlm(const Mat &noisyImage, Mat &denoised, const float sigma, const int searchRadius, const int neighborRadius);
template void swnlm(const Mat &noisyImage, Mat &denoised, const double sigma, const int searchRadius, const int neighborRadius);

template <typename T>
void swnlm(const Mat &noisyImage, Mat &denoised, const T sigma, const int searchRadius, const int neighborRadius)
{
    using SWType = float;

    assert(noisyImage.type() == cv::DataType<T>::type);
    assert(noisyImage.dims == 2);

    const int rows = noisyImage.rows;
    const int cols = noisyImage.cols;

    const SWType alpha = 0.05; // threshold  for accepting null hypothesis. Typically is 0.05 for statistics

    // Pad the edges with a reflection of the outer pixels.
    const int padding = searchRadius + neighborRadius;
    Mat paddedImage;
    copyMakeBorder(noisyImage, paddedImage, padding, padding, padding, padding, BORDER_REFLECT);

    const int paddedFlat[] = {(int)paddedImage.total()};
    paddedImage = paddedImage.reshape(0, 1, paddedFlat);
    T *in = (T *)paddedImage.data;

    const int numNeighbors = (neighborRadius * 2 + 1) * (neighborRadius * 2 + 1);

    vector<SWType> a_vec(numNeighbors / 2 + 1);
    SWType *a = a_vec.data();
    ShapiroWilk::setup(a, numNeighbors);

    const SWType threshold = ShapiroWilk::findThreshold(alpha, numNeighbors);

    const int flatShape[] = {rows * cols};
    denoised.create(1, flatShape, noisyImage.type());
    T *denoisedOut = (T *)denoised.data;

    vector<SWType> diff_vec(numNeighbors);
    SWType *diff = diff_vec.data();

    for (int row = padding; row < rows + padding; row++)
    {
        for (int col = padding; col < cols + padding; col++)
        {
            SWType Wmax = 0;
            SWType avg = 0;
            SWType sumWeights = 0;

            for (int sRow = row - searchRadius; sRow <= row + searchRadius; sRow++)
            {
                for (int sCol = col - searchRadius; sCol <= col + searchRadius; sCol++)
                {
                    if (sRow == row && sCol == col)
                    { // center pixel is skipped
                        continue;
                    }

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

                    SWType w;
                    ShapiroWilk::test(diff, a, numNeighbors, w);

                    if (w > threshold)
                    {
                        // Fail to reject Null hypothesis that it is not normally distributed
                        SWType mean = 0;
                        for (int i = 0; i < numNeighbors; i++)
                        {
                            mean += diff[i];
                        }
                        mean /= numNeighbors;

                        SWType stddev = 0;
                        for (int i = 0; i < numNeighbors; i++)
                        {
                            stddev += (diff[i] - mean) * (diff[i] - mean);
                        }
                        stddev /= numNeighbors;
                        stddev = sqrt(stddev);

                        SWType stderror = stddev / neighborDiam; // Neighborhoods are square, thus sqrt(n) observations is number of rows

                        if (stderror > mean && mean > -stderror &&
                            (1 + stderror > stddev && stddev > 1 - stderror))
                        {
                            Wmax = max(w, Wmax);

                            sumWeights += w;

                            avg += w * in[(2 * padding + cols) * sRow + sCol];
                        }
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
#include "cnlm.hpp"
#include "common.hpp"

#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

template void cnlm(const cv::Mat &noisyImage, cv::Mat &denoised, const uint8_t sigma, const int searchRadius, const int neighborRadius);
template void cnlm(const cv::Mat &noisyImage, cv::Mat &denoised, const int32_t sigma, const int searchRadius, const int neighborRadius);
template void cnlm(const cv::Mat &noisyImage, cv::Mat &denoised, const float sigma, const int searchRadius, const int neighborRadius);
template void cnlm(const cv::Mat &noisyImage, cv::Mat &denoised, const double sigma, const int searchRadius, const int neighborRadius);

template <typename T>
void cnlm(const cv::Mat &noisyImage, cv::Mat &denoised, const T sigma, const int searchRadius, const int neighborRadius)
{
    /**
     * Determine the float precision for intermediate results based on the size of T
     */
    constexpr bool useFloat = sizeof(T) <= sizeof(float);
    using F = typename std::conditional<useFloat, float, double>::type; 

    assert(noisyImage.type() == cv::DataType<T>::type);
    assert(noisyImage.dims == 2);

    const int rows = noisyImage.rows;
    const int cols = noisyImage.cols;
    const F h = 1 * sigma;

    // Pad the edges with a reflection of the outer pixels.
    const int padding = searchRadius + neighborRadius;
    Mat paddedImage;
    copyMakeBorder(noisyImage, paddedImage, padding, padding, padding, padding, BORDER_REFLECT);

    const int paddedFlat[] = {(int)paddedImage.total()};
    paddedImage = paddedImage.reshape(0, 1, paddedFlat);
    T *in = (T *)paddedImage.data;

    vector<F> gaussKernel;
    makeGaussianKernel(gaussKernel, neighborRadius);
    F *kernel = gaussKernel.data();

    const int flatShape[] = {rows * cols};
    denoised.create(1, flatShape, noisyImage.type());
    T *denoisedOut = (T *)denoised.data;

    const int inCols = cols + 2 * padding;

    for (int row = padding; row < rows + padding; row++)
    {
        for (int col = padding; col < cols + padding; col++)
        {
            F sumWeights = 0;
            F val = 0;

            for (int sRow = row - searchRadius; sRow <= row + searchRadius; sRow++)
            {
                for (int sCol = col - searchRadius; sCol <= col + searchRadius; sCol++)
                {
                    F sum = 0;

                    const int neighborDiam = neighborRadius * 2 + 1;
                    for (int y = 0; y < neighborDiam; y++)
                    {
                        for (int x = 0; x < neighborDiam; x++)
                        {
                            const int iNghbrIdx = (2 * padding + cols) * (row + y - neighborRadius) + col + x - neighborRadius;
                            const int jNghbrIdx = (2 * padding + cols) * (sRow + y - neighborRadius) + sCol + x - neighborRadius;
                            const int gaussKernelIdx = y * neighborDiam + x;

                            const F diff = in[iNghbrIdx] - in[jNghbrIdx];

                            sum += diff * diff * kernel[gaussKernelIdx];
                        }
                    }
                    F weight = exp(-sum / (h * h));
                    sumWeights += weight;
                    val += weight * in[sRow * inCols + sCol];
                }
            }
            val /= sumWeights;
            const int denoisedIdx = (row - padding) * cols + col - padding;
            denoisedOut[denoisedIdx] = val;
        }
    }
    const int shape[] = {rows, cols};
    denoised = denoised.reshape(0, 2, shape);
}
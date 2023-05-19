#include "utils.hpp"

#include <opencv2/imgproc.hpp>

#include <utility>

using namespace cv;

double compute_psnr(const Mat &baseImage, const Mat &changedImage)
{
    const int numChannels = baseImage.channels();

    Mat difference = baseImage - changedImage;

    double mse = 0;
    const Scalar mean = cv::mean(difference.mul(difference));
    for (int chnl = 0; chnl < numChannels; chnl++)
    {
        mse += mean[chnl];
    }
    mse /= numChannels;

    const double psnr = 20 * log10(1 / sqrt(mse));

    return psnr;
}

/**
 * Resize an image of size X x Y x P to rows x cols x P
 *
 */
void resize3D(Mat &inputMat, Mat &outputMat, const int rows, const int cols)
{
    const int zDims = inputMat.size[2];
    const int type = inputMat.type();

    const int dims[] = {rows, cols, zDims};
    outputMat.create(3, dims, type);

    Mat slice;

    for (int z = 0; z < zDims; z++)
    {

        mat3toMat2<double>(inputMat, slice, 2, z);

        cv::resize(slice, slice, Size(rows, cols));

        mat2toMat3<double>(slice, outputMat, 2, z);
    }
}

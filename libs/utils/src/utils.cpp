#include "utils.hpp"

#include <opencv2/core.hpp>

using namespace cv;

double computePSNR(const Mat &baseImage, const Mat &changedImage)
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

double computeSSIM(const cv::Mat &baseImage, const cv::Mat &changedImage)
{
    Scalar baseMean, baseStdev;
    Scalar changedMean, changedStdev;

    meanStdDev(baseImage, baseMean, baseStdev);
    meanStdDev(changedImage, changedMean, changedStdev);

    const double baseVariance = baseStdev[0] * baseStdev[0];
    const double changedVariance = changedStdev[0] * changedStdev[0];

    // Default values according to wikipedia
    const double k1 = 0.01;
    const double k2 = 0.03;

    const double c1 = k1 * k1;
    const double c2 = k2 * k2;
    const double c3 = c2 / 2;

    const double luminance = (2 * (baseMean[0] * changedMean[0]) + c1) / (baseMean[0] * baseMean[0] + changedMean[0] * changedMean[0] + c1);

    const double contrast = (2 * baseVariance * changedVariance + c2) / (baseVariance * baseVariance + changedVariance * changedVariance + c2);

    auto baseIt = baseImage.begin<double>();
    auto changedIt = changedImage.begin<double>();
    auto baseEnd = baseImage.end<double>();

    double covariance = 0;

    for (; baseIt != baseEnd; ++baseIt, ++changedIt)
    {
        covariance += (baseIt[0] - baseMean[0]) * (changedIt[0] - changedMean[0]);
    }
    covariance = covariance / (baseImage.rows * baseImage.cols);

    const double structure = (covariance + c3) / (baseStdev[0] * changedStdev[0] + c3);

    return (luminance * contrast * structure + 1) / 2;
}
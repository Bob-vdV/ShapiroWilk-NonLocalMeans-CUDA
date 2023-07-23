#include "utils.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

template double computePSNR<uint8_t>(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);
template double computePSNR<int32_t>(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);
template double computePSNR<float>(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);
template double computePSNR<double>(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);

template <typename T>
double computePSNR(const Mat &baseImage, const Mat &changedImage, const double max)
{
    assert(baseImage.type() == cv::DataType<T>::type);
    assert(changedImage.type() == cv::DataType<T>::type);

    const int numChannels = baseImage.channels();

    Mat difference;
    cv::subtract(baseImage, changedImage, difference, noArray(), CV_64FC1);

    double mse = 0;
    const Scalar mean = cv::mean(difference.mul(difference));
    for (int chnl = 0; chnl < numChannels; chnl++)
    {
        mse += mean[chnl];
    }
    mse /= numChannels;

    const double psnr = 20 * log10(max / sqrt(mse));

    return psnr;
}

template double computeSSIM<uint8_t>(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);
template double computeSSIM<int32_t>(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);
template double computeSSIM<float>(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);
template double computeSSIM<double>(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);

template <typename T>
double computeSSIM(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max)
{
    assert(baseImage.type() == cv::DataType<T>::type);
    assert(changedImage.type() == cv::DataType<T>::type);

    Scalar baseMean, baseStdev;
    Scalar changedMean, changedStdev;

    meanStdDev(baseImage, baseMean, baseStdev);
    meanStdDev(changedImage, changedMean, changedStdev);

    const double baseVariance = baseStdev[0] * baseStdev[0];
    const double changedVariance = changedStdev[0] * changedStdev[0];

    // Default values according to wikipedia
    const double k1 = 0.01;
    const double k2 = 0.03;

    const double c1 = std::pow(k1 * max, 2);
    const double c2 = std::pow(k2 * max, 2);
    const double c3 = c2 / 2;

    const double luminance = (2 * (baseMean[0] * changedMean[0]) + c1) / (baseMean[0] * baseMean[0] + changedMean[0] * changedMean[0] + c1);

    const double contrast = (2 * baseVariance * changedVariance + c2) / (baseVariance * baseVariance + changedVariance * changedVariance + c2);

    auto baseIt = baseImage.begin<T>();
    auto changedIt = changedImage.begin<T>();
    auto baseEnd = baseImage.end<T>();

    double covariance = 0;

    for (; baseIt != baseEnd; ++baseIt, ++changedIt)
    {
        covariance += (baseIt[0] - baseMean[0]) * (changedIt[0] - changedMean[0]);
    }
    covariance = covariance / (baseImage.rows * baseImage.cols);

    const double structure = (covariance + c3) / (baseStdev[0] * changedStdev[0] + c3);

    return (luminance * contrast * structure + 1) / 2;
}

template double computeMSSIM<uint8_t>(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);
template double computeMSSIM<int32_t>(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);
template double computeMSSIM<float>(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);
template double computeMSSIM<double>(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);

template <typename T>
double computeMSSIM(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max)
{
    assert(baseImage.type() == cv::DataType<T>::type);
    assert(changedImage.type() == cv::DataType<T>::type);

    // Default values according to wikipedia
    const double k1 = 0.01;
    const double k2 = 0.03;

    const double c1 = std::pow(k1 * max, 2);
    const double c2 = std::pow(k2 * max, 2);
    const double c3 = c2 / 2;

    const int windowWidth = 8;
    const int n = windowWidth * windowWidth;

    double mssim = 0;

    for (int row = 0; row < baseImage.rows - windowWidth; row++)
    {
        for (int col = 0; col < baseImage.cols - windowWidth; col++)
        {
            double baseMean = 0;
            double changedMean = 0;

            for (int y = row; y < row + windowWidth; y++)
            {
                for (int x = col; x < col + windowWidth; x++)
                {
                    baseMean += baseImage.at<T>(x, y);
                    changedMean += changedImage.at<T>(x, y);
                }
            }
            baseMean /= n;
            changedMean /= n;

            double baseStddev = 0;
            double changedStddev = 0;
            for (int y = row; y < row + windowWidth; y++)
            {
                for (int x = col; x < col + windowWidth; x++)
                {
                    baseStddev += ((double)baseImage.at<T>(x, y) - baseMean) * ((double)baseImage.at<T>(x, y) - baseMean);
                    changedStddev += ((double)changedImage.at<T>(x, y) - changedMean) * ((double)changedImage.at<T>(x, y) - changedMean);
                }
            }
            baseStddev /= n;
            changedStddev /= n;

            baseStddev = sqrt(baseStddev);
            changedStddev = sqrt(changedStddev);

            const double baseVariance = baseStddev * baseStddev;
            const double changedVariance = changedStddev * changedStddev;

            double covariance = 0;

            for (int y = row; y < row + windowWidth; y++)
            {
                for (int x = col; x < col + windowWidth; x++)
                {
                    covariance += ((double)baseImage.at<T>(x, y) - baseMean) * ((double)changedImage.at<T>(x, y) - changedMean);
                }
            }
            covariance = covariance / (windowWidth * windowWidth);

            const double luminance = (2 * (baseMean * changedMean) + c1) / (baseMean * baseMean + changedMean * changedMean + c1);

            const double contrast = (2 * baseVariance * changedVariance + c2) / (baseVariance * baseVariance + changedVariance * changedVariance + c2);

            const double structure = (covariance + c3) / (baseStddev * changedStddev + c3);

            const double ssim = (luminance * contrast * structure);
            mssim += ssim;
        }
    }

    mssim = mssim / ((baseImage.rows - windowWidth) * (baseImage.cols - windowWidth));
    return mssim;
}

template void testNLM(const string filename, const uint8_t sigma, const int searchRadius, const int neighborRadius, NLMFunction<uint8_t> nlmFunction, const bool showImg);
template void testNLM(const string filename, const int32_t sigma, const int searchRadius, const int neighborRadius, NLMFunction<int32_t> nlmFunction, const bool showImg);
template void testNLM(const string filename, const float sigma, const int searchRadius, const int neighborRadius, NLMFunction<float> nlmFunction, const bool showImg);
template void testNLM(const string filename, const double sigma, const int searchRadius, const int neighborRadius, NLMFunction<double> nlmFunction, const bool showImg);

template <typename T, typename Function>
void testNLM(const string filename, const T sigma, const int searchRadius, const int neighborRadius, Function nlmFunction, const bool showImg)
{
    // Ensure that program runs sequentially
    cv::setNumThreads(1);

    // Fix the seed to a constant
    cv::theRNG().state = 42;

    Mat inputImage = imread(filename);
    cvtColor(inputImage, inputImage, COLOR_RGB2GRAY);

    Mat floatImage;
    inputImage.convertTo(floatImage, CV_64FC1);

    Mat noise = floatImage.clone();
    randn(noise, 0, sigma);

    Mat noisyImage = floatImage.clone();
    cv::add(floatImage, noise, noisyImage);

    noisyImage = cv::min(cv::max(noisyImage, 0), 255);

    noisyImage.convertTo(noisyImage, cv::DataType<T>::type);
    floatImage.convertTo(floatImage, cv::DataType<T>::type);

    const double noisyPSNR = computePSNR<T>(floatImage, noisyImage);
    const double noisySSIM = computeSSIM<T>(floatImage, noisyImage);
    const double noisyMSSIM = computeMSSIM<T>(floatImage, noisyImage);

    cout << "Noisy image PSNR: " << noisyPSNR << "\tSSIM: " << noisySSIM << "\tMSSIM: " << noisyMSSIM << '\n';

    const chrono::system_clock::time_point start = chrono::high_resolution_clock::now();

    Mat denoised;
    nlmFunction(noisyImage, denoised, sigma, searchRadius, neighborRadius);

    const chrono::system_clock::time_point end = chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end - start;

    cout << "Finished in " << elapsed_seconds.count() << " seconds\n";

    const double denoisedPSNR = computePSNR<T>(floatImage, denoised);
    const double denoisedSSIM = computeSSIM<T>(floatImage, denoised);
    const double denoisedMSSIM = computeMSSIM<T>(floatImage, denoised);

    cout << "Denoised image PSNR: " << denoisedPSNR << "\tSSIM: " << denoisedSSIM << "\tMSSIM: " << denoisedMSSIM << '\n';

    if (showImg)
    {
        noisyImage.convertTo(noisyImage, CV_8UC1);
        denoised.convertTo(denoised, CV_8UC1);

        imshow("original image", inputImage);
        imshow("noisy image", noisyImage);
        imshow("Denoised", denoised);
        waitKey();
    }
}

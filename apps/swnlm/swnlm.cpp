#include "swnlm.hpp"
#include "utils.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    // Ensure that program runs sequentially
    cv::setNumThreads(1);

    const string filename = "../../images/mandril.tif";
    const double sigma = 0.50;
    const int searchRadius = 10;
    const int neighborRadius = 3;

    Mat inputImage = imread(filename);
    cvtColor(inputImage, inputImage, COLOR_RGB2GRAY);

    Mat floatImage;
    inputImage.convertTo(floatImage, CV_64FC1, 1 / 255.0);

    Mat noise = floatImage.clone();
    randn(noise, 0, sigma);

    Mat noisyImage = floatImage.clone();
    cv::add(floatImage, noise, noisyImage);

    cout << "Noisy image PSNR: " << computePSNR(floatImage, noisyImage) << '\n';

    imshow("original image", inputImage);
    imshow("noisy image", noisyImage);

    const chrono::system_clock::time_point start = chrono::high_resolution_clock::now();

    Mat denoised;
    swnlm(noisyImage, denoised, sigma, searchRadius, neighborRadius);

    const chrono::system_clock::time_point end = chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end - start;

    cout << "Finished in " << elapsed_seconds.count() << " seconds\n";

    const double denoisedPSNR = computePSNR(floatImage, denoised);
    cout << "Denoised image PSNR: " << denoisedPSNR << '\n';

    // denoised.convertTo(denoised, inputImage.type(), 255.0);

    cv::imshow("Denoised", denoised);

    waitKey();
}
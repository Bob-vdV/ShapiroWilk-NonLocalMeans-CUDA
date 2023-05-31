#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "kmeans.hpp"
#include "fastapprox.hpp"
#include "pca.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

/*
TODO:
 - Test PCA
 - Implement Kmeans
 - implement convolutions
 - compute coefficients
 - combine convolutions with coefficients
 - Restructure all the code
*/

/**
 * Input: noisy image of type floating point
 * Output: denoised floating point image
 *
 */
void fasthdnlm(const cv::Mat &noisyImage, cv::Mat &outputImage, const double sigma, const int S, const int windowRadius, const int pcaDims, const int numClusters, const int kMeansImgSize)
{
    // Compute PCA
    cout << "Calculating pca...\n";
    Mat pcaResult;
    computePca(noisyImage, pcaResult, windowRadius, pcaDims);

    // Resize image for Kmeans
    Mat resized;
    resize3D(pcaResult, resized, kMeansImgSize, kMeansImgSize);

    const int newShape[] = {kMeansImgSize * kMeansImgSize, pcaDims};
    resized = resized.reshape(0, 2, newShape);

    // Compute KMeans clusters
    cout << "Clustering data...\n";

    Mat centers;
    kmeansRecursive(resized, centers, numClusters);

    // Filtering
    cout << "Applying filters...\n";

    fastApprox(noisyImage, S, 3.5 * sigma, centers, pcaResult, outputImage);

    cout << cv::mean(outputImage) << '\n';

    cout << "Done\n";
}

int main()
{
    // Ensure that program runs sequentially
    cv::setNumThreads(1);

    const string filename = "../../images/mandril.tif";
    const double sigma = 0.08;//0.08;
    const int S = 10; 
    const int windowRadius = 3;
    const int pcaDims = 25;
    const int numClusters = 31;

    const double sigmaMultiplier = 3.5;

    const int kMeansImgSize = 256;

    Mat inputImage = imread(filename);

    Mat floatImage;
    inputImage.convertTo(floatImage, CV_64FC3, 1 / 255.0);

    Mat noise = floatImage.clone();
    randn(noise, 0, sigma);

    Mat noisyImage = floatImage.clone();
    cv::add(floatImage, noise, noisyImage);

    cout << "Noisy image PSNR: " << compute_psnr(floatImage, noisyImage) << '\n';

    imshow("original image", inputImage);
    imshow("noisy image", noisyImage);

    const chrono::system_clock::time_point start = chrono::high_resolution_clock::now();

    Mat denoised;
    fasthdnlm(noisyImage, denoised, sigma * sigmaMultiplier, S, windowRadius, pcaDims, numClusters, kMeansImgSize);

    const chrono::system_clock::time_point end = chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end - start;

    cout << "Finished in " << elapsed_seconds.count() << " seconds\n";

    const double denoisedPSNR = compute_psnr(floatImage, denoised);
    cout << "Denoised image PSNR: " << denoisedPSNR << '\n';

    denoised.convertTo(denoised, inputImage.type(), 255.0);

    cv::imshow("Denoised", denoised);

    waitKey(0);
}
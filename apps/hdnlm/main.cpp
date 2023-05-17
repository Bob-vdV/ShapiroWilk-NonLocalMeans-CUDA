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

double compute_psnr(const Mat &baseImage, const Mat &changedImage)
{
    Mat difference = baseImage.clone();
    cv::absdiff(changedImage, baseImage, difference);

    cv::imshow("Difference", difference);

    double sum = 0;

    for (int row = 0; row < difference.rows; row++)
    {
        for (int col = 0; col < difference.cols; col++)
        {
            // Repeat for each color channel
            for (size_t channel = 0; channel < 3; channel++)
            {
                const double value = difference.at<cv::Vec3b>(row, col)[channel];
                sum += value * value;
            }
        }
    }
    const double mse = sum / difference.rows / difference.cols;

    const double psnr = 20 * log10(255) - 10 * log10(mse);

    return psnr;
}

/**
 * Resize an image of size X x Y x P to rows x cols x P
 *
 */
void resize3D(Mat &inputMat, Mat &outputMat, const int rows, const int cols)
{
    const int inputRows = inputMat.size[0];
    const int inputCols = inputMat.size[1];
    const int zDims = inputMat.size[2];
    const int size[] = {rows, cols, zDims};
    const int type = inputMat.type();

    outputMat = Mat::zeros(3, size, type);

    const int dims[] = {rows, cols, zDims};
    outputMat.create(3, dims, type);

    for(int z = 0; z < zDims; z++){
        Mat slice(inputRows, inputCols, type);
        for(int row = 0; row < inputRows; row++){
            for(int col = 0; col <inputCols; col++){
                slice.at<double>(row, col) = inputMat.at<double>(row, col, z);
            }
        }

        cv::resize(slice, slice, Size(rows, cols));

        for(int row = 0; row < rows; row++){
            for(int col = 0; col < cols; col++){
                outputMat.at<double>(row, col, z) = slice.at<double>(row, col);
            }
        }
    }
}

int main()
{
    // Ensure that program runs sequentially
    cv::setNumThreads(1);

    const string filename = "../../images/mandril.tif";
    const int sigma = 1; // 100;
    const int S = 10;    // TODO: find out what this is?? Search window?
    const int windowRadius = 3;
    const int pcaDims = 25;
    const int numClusters = 31;

    const int kMeansImgSize = 256;

    Mat inputImage = imread(filename);

    Mat noise = inputImage.clone();
    randn(noise, 0, sigma);

    Mat noisyImage = inputImage.clone();
    cv::add(inputImage, noise, noisyImage);

    cout << "Noisy image PSNR: " << compute_psnr(inputImage, noisyImage) << '\n';

    imshow("original image", inputImage);
    imshow("noisy image", noisyImage);

    // Compute PCA
    Mat inputImageFloat;
    noisyImage.convertTo(inputImageFloat, CV_64FC3, 1 / 255.0);

    imshow("float image", inputImageFloat);

    cout << "Calculating pca...\n";
    Mat pcaResult;
    computePca(inputImageFloat, pcaResult, windowRadius, pcaDims);

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
    Mat denoised;
    fastApprox(inputImageFloat, S, 3.5 * sigma / 256, centers, pcaResult, denoised);

    cv::imshow("Denoised", denoised);

    cout << cv::mean(denoised) << '\n';

    cout << "Done" << endl;

    // TODO compute PSNR

    waitKey(0);
}
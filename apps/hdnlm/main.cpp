#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

/*
TODO:
 - implement PCA
 - implement convolutions
 - compute coefficients
 - combine convolutions with coefficients
 - Restructure all the code
*/

void fastHD_NLM()
{
}

// Should be identical to MatLab circshift() function.
// TODO: test this function
void circularShift(const Mat &inputMat, Mat &outputMat, const int rowOffset, const int ColOffset)
{
    const int rows = inputMat.rows;
    const int cols = inputMat.cols;

    Mat temp = inputMat.clone();

    // Do shift of rows
    for (int i = 0; i < rows; i++)
    {
        temp.row((i + rowOffset + rows) % rows) = inputMat.row(i);
    }

    outputMat = temp.clone();

    // Do shift of columns
    for (int j = 0; j < cols; j++)
    {
        outputMat.col((j + ColOffset + cols) % cols) = temp.col(j);
    }
}

void computePca(const Mat &inputMat, const int windowRadius, const int numDims)
{
    const int rows = inputMat.rows;
    const int cols = inputMat.cols;

    const int numNeighbors = pow(2 * windowRadius + 1, 2);

    // Use vector for three dimensional Matrix
    vector<Mat> spatialKernel(numNeighbors, Mat(rows, cols, inputMat.type()));

    int n = 0;
    for (int i = -windowRadius; i < windowRadius; i++)
    {
        for (int j = -windowRadius; j < windowRadius; j++)
        {
            const size_t dist2 = i * i + j * j;
            const double weight = exp(-dist2 / 2 / (windowRadius / 2));

            Mat C;
            circularShift(inputMat, C, i, j);
            spatialKernel[n] = C * weight;

            n++;
        }
    }

    


    // TODO: compute without openCV function
    // PCA()
}

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

int main()
{
    const string filename = "../../images/mandril.tif";
    const size_t sigma = 100;
    const int windowRadius = 3;
    const int numDimensions = 25;

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
    noisyImage.convertTo(inputImageFloat, CV_32FC3, 1 / 255.0);

    imshow("float image", inputImageFloat);

    Mat normalized = inputImageFloat.clone();
    normalized = normalized - cv::mean(normalized);

    double min, max;
    cv::minMaxIdx(normalized, &min, &max);

    imshow("Normalized image", normalized);

    computePca(normalized, windowRadius, numDimensions);

    //    cv::PCA pca(normalized, Mat(), cv::PCA::DATA_AS_COL, numDimensions);

    // TODO

    waitKey(0);
}
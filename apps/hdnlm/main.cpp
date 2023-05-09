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

    Mat inputImage = imread(filename);

    Mat noise = inputImage.clone();
    randn(noise, 0, sigma);

    Mat noisyImage = inputImage.clone();
    cv::add(inputImage, noise, noisyImage);

    cout << "Noisy image PSNR: " << compute_psnr(inputImage, noisyImage) << '\n';

    imshow("original image", inputImage);
    imshow("noisy image", noisyImage);
    waitKey(0);

    // Compute PCA
    //TODO


}
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

void kmeans(Mat &inputMat, Mat &outputMat, int clusters){
    

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

void computePca(const Mat &inputMat, const int windowRadius, const int numDims, Mat &outputMat)
{
    const int rows = inputMat.rows;
    const int cols = inputMat.cols;
    const int channels = inputMat.channels();
    const int numNeighbors = pow(2 * windowRadius + 1, 2);


    int dims[3] = {rows, cols, numNeighbors};

    Mat spatialKernel(3, dims, inputMat.type());

    int n = 0;
    Mat C;

    for (int i = -windowRadius; i < windowRadius; i++)
    {
        for (int j = -windowRadius; j < windowRadius; j++)
        {
            const int dist2 = i * i + j * j;
            const double weight = exp(-dist2 / 2.0 / (windowRadius / 2.0));

            circularShift(inputMat, C, i, j);

            Range ranges[3] = {
                Range::all(),
                Range::all(),
                Range(n, n+1)
            };


            /* Hacky workaround for slicing a 3d matrix by reference:
             * slice it, convert C to rows x cols x 1 and then add C to
             * the slice instead of setting it. This keeps the reference.
            */
            Mat slice = spatialKernel(ranges);

            int dims3d[3] = {rows, cols, 1};

            Mat temp(3, dims3d, C.type());
            temp.data = C.data;

            slice += temp * weight;
            n++;
        }
    }

    Mat flattened;

    int newShape[2] = {rows * cols, numNeighbors * channels};
    flattened = spatialKernel.reshape(1, 2, newShape);

    flattened = flattened - cv::mean(flattened);

    cv::PCA pca(flattened, noArray(), cv::PCA::DATA_AS_COL, numDims);

    cv::transpose(pca.eigenvectors, outputMat);

    outputMat = outputMat.reshape(0, {rows, cols, numDims});
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
    const int pcaDims = 25;

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

    Mat normalized = inputImageFloat.clone();
    normalized = normalized - cv::mean(normalized);

    imshow("Normalized image", normalized);

    cout << "Calculating pca...\n";
    Mat pcaResult;
    computePca(normalized, windowRadius, pcaDims, pcaResult);


    //TODO: Resize??

    // Compute KMeans clusters


    // TODO

    //waitKey(0);
}  
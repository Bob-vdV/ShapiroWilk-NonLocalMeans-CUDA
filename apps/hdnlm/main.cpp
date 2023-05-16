#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "kmeans.hpp"
#include "fastapprox.hpp"

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

void computePca(const Mat &inputMat, Mat &outputMat, const int windowRadius, const int numDims)
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
                Range(n, n + 1)};

            /* Hacky workaround for slicing a 3d matrix by reference:
             * slice it, convert C to rows x cols x 1 and then add C to
             * the slice instead of setting it. This keeps the reference.
             */
            Mat slice = spatialKernel(ranges);

            int dims3d[] = {rows, cols, 1};

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

void resize3D(const Mat &inputMat, Mat &outputMat, const int rows, const int cols)
{
    const int zDims = inputMat.size[2];
    const int size[] = {rows, cols, zDims};
    const int type = inputMat.type();

    outputMat = Mat::zeros(3, size, type);

    for (int z = 0; z < zDims; z++)
    {
        Range ranges[] = {
            Range::all(),
            Range::all(),
            Range(z, z + 1)};

        Mat inputSlice3D = inputMat(ranges);

        // Create 2D Mat that can be resized properly.
        Mat inputSlice2D(inputMat.size[0], inputMat.size[1], type);
        inputSlice2D.data = inputSlice3D.data;

        Mat outputSlice2D(rows, cols, type);
        cv::resize(inputSlice2D, outputSlice2D, Size(rows, cols));

        int sliceSize[] = {rows, cols, 1};
        Mat outputSlice3D(3, sliceSize, type);
        outputSlice3D.data = outputSlice2D.data;

        outputMat(ranges) += outputSlice3D;
    }
}

int main()
{
    // Ensure that program runs sequentially
    cv::setNumThreads(1);

    const string filename = "../../images/mandril.tif";
    const size_t sigma = 100;
    const int S = 10; // TODO: find out what this is?? Search window?
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

    Mat normalized = inputImageFloat.clone();
    normalized = normalized - cv::mean(normalized);

    imshow("Normalized image", normalized);

    cout << "Calculating pca...\n";
    Mat pcaResult;
    computePca(normalized, pcaResult, windowRadius, pcaDims);

    // Resize image for Kmeans
    Mat resized;
    resize3D(pcaResult, resized, kMeansImgSize, kMeansImgSize);

    int newShape[] = {kMeansImgSize * kMeansImgSize, pcaDims};
    resized = resized.reshape(0, 2, newShape);

    // Compute KMeans clusters
    cout << "Clustering data...\n";

    Mat centers;
    kmeansRecursive(resized, centers, numClusters);


    //Filtering
    Mat denoised;
    fastApprox(inputImageFloat, S, 3.5 * sigma / 256, centers, pcaResult, denoised);

    cv::imshow("Denoised", denoised);

    cout << "Done" << endl;

    waitKey(0);

}
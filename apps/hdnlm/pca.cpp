#include "pca.hpp"

// TODO remove
#include <iostream>

using namespace std;
using namespace cv;

// Should be identical to MatLab circshift() function.
void circularShift(const Mat &inputMat, Mat &outputMat, const int rowOffset, const int ColOffset)
{
    const int rows = inputMat.rows;
    const int cols = inputMat.cols;
    const int type = inputMat.type();

    Mat temp = Mat::zeros(rows, cols, type);

    // Do shift of rows
    for (int i = 0; i < rows; i++)
    {
        temp.row((i + rowOffset + rows) % rows) += inputMat.row(i);
    }

    outputMat = Mat::zeros(rows, cols, type);

    // Do shift of columns
    for (int j = 0; j < cols; j++)
    {
        outputMat.col((j + ColOffset + cols) % cols) += temp.col(j);
    }
}

void computePca(const Mat &inputMat, Mat &outputMat, const int windowRadius, const int numDims)
{
    const int rows = inputMat.rows;
    const int cols = inputMat.cols;
    const int channels = inputMat.channels();
    const int numNeighbors = pow(2 * windowRadius + 1, 2);
    const int type = inputMat.type();

    int dims[3] = {rows, cols, numNeighbors};

    Mat spatialKernel(3, dims, type);

    int n = 0;
    Mat C;

    Mat normalized = inputMat - cv::mean(inputMat);

    for (int i = -windowRadius; i < windowRadius; i++)
    {
        for (int j = -windowRadius; j < windowRadius; j++)
        {
            const int dist2 = i * i + j * j;
            const double weight = exp(-dist2 / 2.0 / (windowRadius / 2.0));

            circularShift(normalized, C, i, j);

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

    flattened -= cv::mean(flattened);

    Mat eigenvalues, eigenVectors;
    cv::eigen(flattened.t() * flattened, eigenvalues, eigenVectors);

    const Range eigenRange[] = {Range::all(), Range(0, numDims)};

    flattened = flattened * eigenVectors(eigenRange);
    outputMat = flattened.reshape(0, {rows, cols, numDims});
}
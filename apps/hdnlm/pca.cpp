#include "pca.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

// TODO remove
#include <iostream>

// Should be identical to MatLab circshift() function.
// TODO do in one shot
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

    for (int i = -windowRadius; i <= windowRadius; i++)
    {
        for (int j = -windowRadius; j <= windowRadius; j++)
        {
            const int dist2 = i * i + j * j;
            const double weight = exp(-dist2 / 2.0 / (windowRadius / 2.0));

            circularShift(normalized, C, i, j);

            Mat slice = C * weight;

            mat2toMat3<cv::Vec3d>(slice, spatialKernel, 2, n);

            n++;
        }
    }

    // TODO: remove block
    {
        double min = INFINITY, max = -INFINITY;
        for (auto it = spatialKernel.begin<Vec3d>(); it != spatialKernel.end<Vec3d>(); ++it)
        {
            for (int i = 0; i < 3; i++)
            {
                min = cv::min(min, it[0][i]);
                max = cv::max(max, it[0][i]);
            }
        }

        cout << "SpatialKernel: " << cv::mean(spatialKernel) << '\t' << min << '\t' << max << '\n';
    }
    //------

    int newShape[2] = {rows * cols, numNeighbors * channels};
    Mat flattened = spatialKernel.reshape(1, 2, newShape);

    Mat means;
    cv::reduce(flattened, means, 0, REDUCE_AVG);

    for (int row = 0; row < flattened.rows; row++)
    {
        flattened.row(row) -= means;
    }

    // TODO: remove block
    {
        double min, max;
        min = INFINITY;
        max = -INFINITY;
        for (auto it = flattened.begin<double>(); it != flattened.end<double>(); ++it)
        {
            min = cv::min(min, it[0]);
            max = cv::max(max, it[0]);
        }

        cout << "flattened: " << cv::mean(flattened) << '\t' << min << '\t' << max << '\n';
    }

    //------

    Mat eigenvalues, eigenVectors;
    cv::eigen(flattened.t() * flattened, eigenvalues, eigenVectors);

    const Range eigenRange[] = {Range::all(), Range(0, numDims)};

    eigenVectors = eigenVectors(eigenRange);

    flattened = flattened * eigenVectors;

        // TODO: remove block
    {
        double min, max;
        min = INFINITY;
        max = -INFINITY;
        for (auto it = flattened.begin<double>(); it != flattened.end<double>(); ++it)
        {
            min = cv::min(min, it[0]);
            max = cv::max(max, it[0]);
        }

        cout << "flattened after matmul: " << cv::mean(flattened) << '\t' << min << '\t' << max << '\n';
    }

    //------




    outputMat = flattened.reshape(0, {rows, cols, numDims});

    // TODO: remove block
    double min, max;
    min = INFINITY;
    max = -INFINITY;
    for (auto it = outputMat.begin<double>(); it != outputMat.end<double>(); ++it)
    {
        min = cv::min(min, it[0]);
        max = cv::max(max, it[0]);
    }

    cout << "PCA output: " << cv::mean(outputMat) << '\t' << min << '\t' << max << '\n';

    Mat test = eigenVectors(eigenRange);

    min = INFINITY;
    max = -INFINITY;
    for (auto it = test.begin<double>(); it != test.end<double>(); ++it)
    {
        min = cv::min(min, it[0]);
        max = cv::max(max, it[0]);
    }

    cout << "Eigenvectors: " << cv::mean(outputMat) << '\t' << min << '\t' << max << '\n';

    cout << eigenVectors << '\n';

    cout << eigenVectors.at<double>(0, 0) << '\n';
    cout << eigenVectors.at<double>(1, 0) << '\n';


    //--
}
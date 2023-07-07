#include "common.hpp"

#include <cmath>

using namespace std;

void makeGaussianKernel(vector<double> &gaussKernel, const int neighborRadius)
{
    const int rows = neighborRadius * 2 + 1;
    const int cols = neighborRadius * 2 + 1;

    const int stdev = 1;
    const int middle = neighborRadius;

    gaussKernel.resize(rows * cols);

    double *kernel = gaussKernel.data();

    double sum = 0;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            const int rowDist = row - middle;
            const int colDist = col - middle;

            kernel[row * cols + col] = exp(((rowDist * rowDist) + (colDist * colDist)) / (-2.0 * stdev * stdev));
            sum += kernel[row * cols + col];
        }
    }

    for (int elem = 0; elem < rows * cols; elem++)
    {
        kernel[elem] /= sum;
    }
}
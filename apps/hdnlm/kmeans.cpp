#include "kmeans.hpp"

#include <iostream>

using namespace std;
using namespace cv;

/**
 * Prerequisites:
 * inputMat has size [rows x 2]
 */
void minPerColumn(const Mat &inputMat, Mat &mins, Mat &minIndices)
{
    const int rows = inputMat.rows;
    const int type = inputMat.type();
    assert(inputMat.cols == 2);

    mins = Mat(rows, 1, type);
    for (int r = 0; r < rows; r++)
    {
        const double first = inputMat.at<double>(r, 0);
        const double second = inputMat.at<double>(r, 1);

        if (first <= second)
        {
            minIndices.at<double>(r, 0) = 0;
            mins.at<double>(r, 0) = first;
        }
        else
        {
            minIndices.at<double>(r, 0) = 1;
            mins.at<double>(r, 0) = second;
        }
    }
}

/**
 * Slice a matrix where the bools Mat is nonzero.
 * Output has all rows from inputMat where respective bool value is true.
 * 
*/
void sliceMat(const Mat &inputMat, const Mat &bools, Mat &slice)
{
    const int inputRows = inputMat.rows;

    slice = Mat();

    for (int iRow = 0; iRow < inputRows; iRow++)
    {
        if (bools.at<uint8_t>(iRow, 0))
        {
            slice.push_back(inputMat.row(iRow));
        }
    }
}
/**
 * Index of the maximum in the given Mat
*/
int maxIdx(const Mat &inputMat)
{
    int maxIdx = 0;
    double max = inputMat.at<double>(0, 0);

    for (int i = 1; i < inputMat.rows; i++)
    {
        double val = inputMat.at<double>(i, 0);
        if (max < val)
        {
            val = max;
            maxIdx = i;
        }
    }
    return maxIdx;
}

/**
 * Split the inputMat into 2 clusters and return their centers.
 * Returns: gIdx, C, dist, clust
 */
void kmeansCluster(const Mat &inputMat, Mat &gIdx, Mat &C, Mat &dist, Mat &clust)
{
    const int rows = inputMat.rows;
    const int cols = inputMat.cols;
    const int type = inputMat.type();

    C = Mat::zeros(2, cols, type);

    Mat squared;
    cv::multiply(inputMat, inputMat, squared);


    Mat Y;

    cv::reduce(squared, Y, 1, cv::REDUCE_SUM);

    cout << cv::mean(inputMat)[0] << '\t' << cv::mean(squared)[0] << '\t' << cv::mean(Y)[0] << '\n';


    int minIdx, maxIdx;

    cv::minMaxIdx(Y, NULL, NULL, &minIdx, &maxIdx);

    inputMat.row(minIdx).copyTo(C.row(0));
    inputMat.row(maxIdx).copyTo(C.row(1));

    Mat g0 = Mat::ones(rows, 1, type);
    gIdx = Mat::zeros(rows, 1, type);
    Mat D = Mat::zeros(rows, 2, type);

    Mat mins;

    // While g0 != gIdx
    while (cv::countNonZero(g0 != gIdx) != 0)
    {
        g0 = gIdx;
        // Loop for each centroid
        for (int t = 0; t < 2; t++)
        {
            Mat temp(rows, cols, type);
            for (int r = 0; r < rows; r++)
            {
                Mat vec = inputMat.row(r) - C.row(t);
                vec.copyTo(temp.row(r));
            }

            cv::multiply(temp, temp, temp);

            Mat sum;
            cv::reduce(temp, sum, 1, cv::REDUCE_SUM);

            sum.copyTo(D.col(t));
        }

        // Partition data to closest centroids
        minPerColumn(D, mins, gIdx);

        clust = Mat(2, 1, CV_32SC1);

        for (int t = 0; t < 2; t++)
        {
            for (int col = 0; col < cols; col++)
            {
                double sum = 0;
                for (int row = 0; row < rows; row++)
                {
                    if ((int)gIdx.at<double>(row, 0) == t)
                    {
                        sum += inputMat.at<double>(row, col);
                    }
                }
                const double mean = sum / rows;
                C.at<double>(t, col) = mean;
            }
        }

        // Count number of points in each cluster
        clust.at<int>(0, 0) = rows - cv::sum(gIdx)[0];
        clust.at<int>(1, 0) = cv::sum(gIdx)[0];
    }
    dist = Mat(2, 1, type);

    for (int t = 0; t < 2; t++)
    {
        double sum = 0;
        for (int row = 0; row < rows; row++)
        {
            if (gIdx.at<double>(row, 0) == t)
            {
                sum += mins.at<double>(row, 0);
            }
        }

        dist.at<double>(t, 0) = sum;
    }
}

void kmeansRecursive(const Mat &inputMat, Mat &center, int clusters)
{
    const int dims = inputMat.cols;
    const int type = inputMat.type();

    int K = 0;

    Mat minCenter, centerTemp, newDist, newClust;
    kmeansCluster(inputMat, minCenter, centerTemp, newDist, newClust);

    center = Mat::zeros(clusters, dims, type);
    Mat var = Mat::zeros(clusters, 1, type);
    K++;

    centerTemp.row(0).copyTo(center.row(0));
    centerTemp.row(1).copyTo(center.row(1));

    var.at<double>(0, 0) = newDist.at<double>(0, 0);
    var.at<double>(1, 0) = newDist.at<double>(1, 0);

    int maxIndex = maxIdx(var);
    while (K < clusters)
    {
  Mat label;
        Mat slice;
        sliceMat(inputMat, minCenter == maxIndex, slice);
        kmeansCluster(slice, label, centerTemp, newDist, newClust);

        if (newClust.at<double>(0, 0) == 0 || newClust.at<double>(1, 0) == 0)
        {
            var.at<double>(maxIndex, 0) = 0;
        }
        else
        {
            K++;

            // Populating new cluster centers
            centerTemp.row(0).copyTo(center.row(maxIndex));
            centerTemp.row(1).copyTo(center.row(K));

            // Populating new cluster indices vector
            for (int r = 0; r < label.rows; r++)
            {
                if (label.at<double>(r, 0) == 0)
                {
                    label.at<double>(r, 0) = maxIndex;
                }
                else
                {
                    label.at<double>(r, 0) = K;
                }
            }
            int lRow = 0;
            for (int j = 0; j < minCenter.rows; j++)
            {
                if (minCenter.at<double>(j, 0) == maxIndex)
                {
                    minCenter.at<double>(j, 0) = label.at<double>(lRow, 0);
                    lRow++;
                }
            }

            var.at<double>(maxIndex, 0) = newDist.at<double>(0, 0);
            var.at<double>(K, 0) = newDist.at<double>(1, 0);
        }

        if (cv::countNonZero(var) == var.size[0])
        {
            break;
        }
        maxIndex = maxIdx(var);
    }

    // TODO
}
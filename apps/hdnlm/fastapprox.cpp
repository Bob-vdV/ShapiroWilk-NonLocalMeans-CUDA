#include "fastapprox.hpp"
#include "filter.hpp"

#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void fastApprox(const Mat &inputImage, const int S, const double h, Mat &center, const Mat &guideImage, Mat &outputImage)
{
    const int guidedDims = guideImage.size[2];

    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const int type = inputImage.type();

    Mat B = Mat::zeros(rows, cols, type);

    const int clusters = center.rows;

    // Forming intermediate images and coefficients
    Mat C1 = Mat::zeros(clusters, clusters, CV_64FC1);

    for (int i = 0; i < clusters; i++)
    {
        C1.at<double>(i, i) = 1; // TODO: CHECK IF THIS is correct

        for (int j = i; j < clusters; j++)
        {
            Mat diff = center.row(i) - center.row(j);
            diff = diff.mul(diff); // element wise square
            C1.at<double>(i, j) = -cv::sum(diff)[0] / (2 * h * h);
            C1.at<double>(j, i) = C1.at<double>(i, j);
        }
    }

    Mat C1chan = C1.inv(cv::DECOMP_SVD);

    // TODO: Probably very slow, needs to be improved.
    int dims[] = {rows, cols, clusters};
    Mat W = Mat::zeros(3, dims, type);

    for (int i = 0; i < clusters; i++)
    {
        Mat resized = center.row(i).reshape(0, {guidedDims, 1});

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                double sum = 0;

                for (int j = 0; j < clusters; j++)
                {
                    const double temp = guideImage.at<double>(row, col, j) - resized.at<double>(j, 0);
                    sum += temp * temp;
                }
                W.at<double>(row, col, i) = sum;
            }
        }
    }

    W = -W / (2 * h * h);

    cv::exp(W, W);
    Mat Wb = Mat::zeros(rows, cols, type);

    // Box filtering using O(1) convolutions
    for (int i = 0; i < clusters; i++)
    {
        Mat Wt(rows, cols, type);
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                int sum = 0;
                for (int cluster = 0; cluster < clusters; cluster++)
                {
                    sum += W.at<double>(row, col, cluster) * C1chan.at<double>(i, cluster);
                }
                Wt.at<double>(row, col) = sum;
            }
        }

        Range ranges[] = {Range::all(), Range::all(), Range(i, i + 1)};
        Mat WSlice = W(ranges);

        WSlice.copySize(Mat(rows, cols, type));

        Mat box;
        boxFilter(WSlice, S, box);

        Wb += Wt.mul(box);

        Mat multiplied = inputImage.clone();

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                multiplied.at<cv::Vec3d>(row, col) *= W.at<double>(row, col, i);
            }
        }

        Mat BBox;
        boxFilter(multiplied, S, BBox);

        B += Wt.mul(BBox);
    }
    cv::divide(B, Wb, outputImage);

    //cout << cv::mean(B) << '\t' << cv::mean(Wb) << '\n';
    //cout << outputImage.at<Vec3d>(0,0) << '\n';
    //cout << cv::mean(outputImage) << '\n';
}

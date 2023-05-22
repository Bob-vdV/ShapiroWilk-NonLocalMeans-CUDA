#include "fastapprox.hpp"
#include "filter.hpp"
#include "utils.hpp"

#include "opencv2/imgproc.hpp"

// TODO REMOVE
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void fastApprox(const Mat &inputImage, const int S, const double h, Mat &center, const Mat &guideImage, Mat &outputImage)
{
    const int guidedDims = guideImage.size[2];

    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const int inputType = inputImage.type(); // Multiple channels
    const int guideType = guideImage.type(); // 1 channel

    Mat B = Mat::zeros(rows, cols, inputType);

    const int clusters = center.rows;

    // Forming intermediate images and coefficients
    Mat C1 = Mat::zeros(clusters, clusters, guideType);

    for (int i = 0; i < clusters; i++)
    {
        C1.at<double>(i, i) = 1;

        for (int j = i+1; j < clusters; j++)
        {
            Mat diff = center.row(i) - center.row(j);
            diff = diff.mul(diff); // element wise square
            C1.at<double>(i, j) = exp(-cv::sum(diff)[0] / (2 * h * h));
            C1.at<double>(j, i) = C1.at<double>(i, j);
        }
    }

    const Mat C1chan = C1.inv(cv::DECOMP_SVD);

    cout << center.size << '\n';
    cout << guideImage.size << '\n';
    cout << "C1: " << cv::mean(C1)[0] << '\n';

    double min = 0, max = 0;
    int minIdx = 0, maxIdx =0;
    cv::minMaxIdx(C1, &min, &max, &minIdx, &maxIdx);
    cout << min << '\t' << max << '\t' << minIdx << '\t' << maxIdx << '\n';

    cout << "C1Chan: " << cv::mean(C1chan) << '\n';


    int dims[] = {rows, cols, clusters};
    Mat W = Mat::zeros(3, dims, CV_64FC1);

    for (int i = 0; i < clusters; i++)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                double sum = 0;

                for (int gd = 0; gd < guidedDims; gd++)
                {
                    const double temp = guideImage.at<double>(row, col, gd) - center.at<double>(i, gd);
                    sum += temp * temp;
                }
                W.at<double>(row, col, i) = exp(-sum / (2 * h * h));
            }
        }
    }

    cout << "W: " << cv::mean(W) << '\n';
    minMaxIdx(W, &min, &max, &minIdx, &maxIdx);
    cout << min << '\t' << max << '\t' << minIdx << '\t' << maxIdx << '\n';

    Mat Wb = Mat::zeros(rows, cols, guideType);

    // Box filtering using O(1) convolutions
    for (int i = 0; i < clusters; i++)
    {
        Mat Wt(rows, cols, guideType);
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

        cout << "Wt " << cv::mean(Wt) << '\n';

        Mat WSlice;
        mat3toMat2<double>(W, WSlice, 2, i);

        cout << "Slice " << cv::mean(WSlice)[0] << '\n';

        Mat box;
        boxFilter(WSlice, S, box);

        Wb += Wt.mul(box);

        cout << "Wb " << cv::mean(Wb) << '\n';

        Mat multiplied = inputImage.clone();

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                multiplied.at<Vec3d>(row, col) *= W.at<double>(row, col, i);
            }
        }

        Mat BBox;
        boxFilter(multiplied, S, BBox);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                B.at<Vec3d>(row, col) += Wt.at<Vec3d>(row, col) * BBox.at<double>(row, col);
            }
        }
    }

    imshow("B", B);
    imshow("Wb", Wb);

    outputImage.create(rows, cols, inputType);

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            outputImage.at<Vec3d>(row, col) = B.at<Vec3d>(row, col) / Wb.at<double>(row, col);

            /*if(Wb.at<double>(row, col) == 0.0){
                cout << "0 at " << row << ", " << col << '\n';
            }*/
        }
    }

    cout << cv::mean(B) << '\t' << cv::mean(Wb)[0] << '\n';
    // cout << outputImage.at<Vec3d>(0,0) << '\n';
    // cout << cv::mean(outputImage) << '\n';
}

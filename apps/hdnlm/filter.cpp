#include "filter.hpp"

//TODO REMOVE
#include <iostream>

using namespace std;
using namespace cv;

void box1D(const Mat &inputMat, const int S, Mat &outputMat)
{
    const int rows = inputMat.rows;
    const int cols = inputMat.cols;
    const int type = inputMat.type();

    assert(type == CV_64FC1);
    assert(rows > 0 && rows > S);
    assert(cols == 1);

    outputMat.create(rows, 1, type);

    Range ranges[] = {
        Range(0, S + 1),
        Range::all()};

    outputMat.at<double>(0, 0) = cv::sum(inputMat(ranges))[0];

    for (int i = 1; i < S + 1; i++)
    {
        outputMat.at<double>(i, 0) = outputMat.at<double>(i - 1, 0) + inputMat.at<double>(i + S, 0);
    }
    for (int i = S + 1; i < rows - S; i++)
    {
        outputMat.at<double>(i, 0) = outputMat.at<double>(i - 1, 0) + inputMat.at<double>(i + S, 0) - inputMat.at<double>(i - (S + 1), 0);
    }
    for (int i = rows - S; i < rows; i++)
    {
        outputMat.at<double>(i, 0) = outputMat.at<double>(i - 1, 0) - inputMat.at<double>(i - S + 1, 0);
    }
}

void box2D(const Mat &inputMat, const int S, Mat &outputMat)
{
    assert(inputMat.type() == CV_64FC1);

    const int rows = inputMat.rows;
    const int cols = inputMat.cols;
    const int type = inputMat.type();

    Mat outputTemp(rows, cols, type);

    for (int row = 0; row < rows; row++)
    {
        Mat inputRow = inputMat.row(row).t();

        Mat boxRow;
        box1D(inputRow, S, boxRow);

        boxRow = boxRow.t();

        boxRow.copyTo(outputTemp.row(row));
    }

    outputMat.create(rows, cols, type);
    for (int col = 0; col < cols; col++)
    {
        Mat boxCol;
        box1D(outputTemp.col(col), S, boxCol);

        boxCol.copyTo(outputMat.col(col));
    }
}

void boxFilter(const Mat &inputMat, const int S, Mat &outputMat)
{
    const int numChannels = inputMat.channels();

    Mat inputChannels[numChannels];
    Mat outputChannels[numChannels];

    cv::split(inputMat, inputChannels);

    for (int chnl = 0; chnl < numChannels; chnl++)
    {
        box2D(inputChannels[chnl], S, outputChannels[chnl]);
        // cout << inputChannels[chnl].at<double>(0,0) << '\n';
    }
    cv::merge(outputChannels, numChannels, outputMat);

    // cout << "out " <<  outputMat.at<Vec3d>(0,0) << '\n';
}

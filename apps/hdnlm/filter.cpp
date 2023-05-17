#include "filter.hpp"

#include <iostream>

using namespace std;
using namespace cv;

void box1D(const Mat &inputMat, const int S, Mat &outputMat)
{
    const int rows = inputMat.rows;
    const int type = inputMat.type();

    outputMat = Mat::zeros(rows, 1, type);

    Range ranges[] = {
        Range(0, S + 1),
        Range::all()};

    outputMat.at<double>(0, 0) = cv::sum(inputMat(ranges))[0];

    for (int i = 1; i < S + 1; i++)
    {
        outputMat.at<double>(i, 0) = outputMat.at<double>(i - 1, 0) + inputMat.at<double>(i + S);
    }
    for (int i = S + 2; i < rows - S; i++)
    {
        outputMat.at<double>(i, 0) = outputMat.at<double>(i - 1, 0) + inputMat.at<double>(i + S) - inputMat.at<double>(i - (S + 1), 0);
    }
    for (int i = rows - S + 1; i < rows; i++)
    {
        outputMat.at<double>(i, 0) = outputMat.at<double>(i - 1, 0) - inputMat.at<double>(i - S + 1, 0);
    }
}

void box2D(const Mat &inputMat, const int S, Mat &outputMat)
{
    const int rows = inputMat.rows;
    const int cols = inputMat.cols;
    const int type = inputMat.type();

    Mat outputTemp(0, cols, type);

    for (int row = 0; row < rows; row++)
    {
        Mat boxRow;

        Mat inputRow;
        const int inShape[] = {cols, 1};

        inputRow = inputMat.row(row).reshape(0, 2, inShape);

        box1D(inputRow, S, boxRow);

        const int outShape[] = {1, cols};
        boxRow = boxRow.reshape(0, 2, outShape);

        cv::vconcat(outputTemp, boxRow, outputTemp);
    }

    outputMat = Mat(rows, 0, type);

    for (int col = 0; col < cols; col++)
    {
        Mat boxCol;
        box1D(outputTemp.col(col), S, boxCol);

        cv::hconcat(outputMat, boxCol, outputMat);
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
        //cout << inputChannels[chnl].at<double>(0,0) << '\n';
    }
    cv::merge(outputChannels, numChannels, outputMat);

    //cout << "out " <<  outputMat.at<Vec3d>(0,0) << '\n';
}

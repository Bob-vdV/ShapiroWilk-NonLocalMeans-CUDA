#include "swnlm.hpp"
#include "swilk.hpp"

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

void swnlm(const Mat &noisyImage, Mat &denoised, const double sigma, const int searchRadius, const int neighborRadius)
{
    assert(noisyImage.type() == CV_64FC1);
    assert(noisyImage.dims == 2);

    const int rows = noisyImage.rows;
    const int cols = noisyImage.cols;

    double alpha = 0.05; // threshold  for accepting null hypothesis. Typically is 0.05 for statistics

    // Pad the edges with a reflection of the outer pixels.
    const int padding = searchRadius + neighborRadius;
    Mat paddedImage;
    copyMakeBorder(noisyImage, paddedImage, padding, padding, padding, padding, BORDER_REFLECT);

    const int shape[] = {rows, cols};
    denoised.create(2, shape, noisyImage.type());

    ShapiroWilk swilk(pow((neighborRadius * 2 + 1), 2));

    for (int row = padding; row < rows + padding; row++)
    {
        for (int col = padding; col < cols + padding; col++)
        {
            double Wmax = 0;
            double avg = 0;
            double sumWeights = 0;

            const Range rangesI[] = {Range(row - neighborRadius, row + neighborRadius + 1), Range(col - neighborRadius, col + neighborRadius + 1)};
            const Mat neighborhoodI = paddedImage(rangesI);

            for (int sRow = row - searchRadius; sRow <= row + searchRadius; sRow++)
            {
                for (int sCol = col - searchRadius; sCol <= col + searchRadius; sCol++)
                {
                    if (sRow == row && sCol == col)
                    { // center pixel is skipped
                        continue;
                    }

                    const Range rangesJ[] = {Range(sRow - neighborRadius, sRow + neighborRadius + 1), Range(sCol - neighborRadius, sCol + neighborRadius + 1)};
                    const Mat neighborhoodJ = paddedImage(rangesJ);

                    Mat difference = neighborhoodI - neighborhoodJ;
                    Mat normalized = difference / (sqrt(2) * sigma);

                    assert(normalized.isContinuous());
                    std::vector<double> array;
                    array.assign((double *)normalized.data, (double *)normalized.data + normalized.total());

                    double w, pw;
                    swilk.test(array, w, pw);

                    // cout << w << '\t' << pw << '\n';
                    Scalar mean, stdev;
                    cv::meanStdDev(normalized, mean, stdev);

                    double stderror = stdev[0] / normalized.rows; // Neighborhoods are square, thus sqrt(n) observations is number of rows

                    if (stderror > mean[0] && mean[0] > -stderror &&
                        (1 + stderror > stdev[0] && stdev[0] > 1 - stderror) &&
                        pw > alpha)
                    {
                        Wmax = max(w, Wmax);

                        sumWeights += w;
                        avg += w * paddedImage.at<double>(sRow, sCol);
                    }
                }
            }
            avg += Wmax * paddedImage.at<double>(row, col);
            sumWeights += Wmax;
            if (sumWeights > 0)
            {
                denoised.at<double>(row - padding, col - padding) = avg / sumWeights;
            }
            else
            {
                denoised.at<double>(row - padding, col - padding) = paddedImage.at<double>(row, col);
            }
        }
    }
}
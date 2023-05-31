#include "swilk.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>

using namespace std;
using namespace cv;

double compute_psnr(const Mat &baseImage, const Mat &changedImage)
{
    const int numChannels = baseImage.channels();

    Mat difference = baseImage - changedImage;

    double mse = 0;
    const Scalar mean = cv::mean(difference.mul(difference));
    for (int chnl = 0; chnl < numChannels; chnl++)
    {
        mse += mean[chnl];
    }
    mse /= numChannels;

    const double psnr = 20 * log10(1 / sqrt(mse));

    return psnr;
}

void swnlm(const Mat &noisyImage, Mat &denoised, const int searchRadius, const int neighborRadius)
{
    assert(noisyImage.type() == CV_64FC1);
    assert(noisyImage.dims == 2);

    const int rows = noisyImage.rows;
    const int cols = noisyImage.cols;

    double alpha = 0.05; // Parameter / threshold  for accepting null hypothesis

    // Pad the edges with a reflection of the outer pixels.
    const int padding = searchRadius + neighborRadius;
    Mat paddedImage;
    copyMakeBorder(noisyImage, paddedImage, padding, padding, padding, padding, BORDER_REFLECT);

    const int shape[] = {rows, cols};
    denoised.create(2, shape, noisyImage.type());

    // TODO: determine sigma automatically
    double sigma = 0.10;

    for (int row = padding; row < rows + padding; row++)
    {
        cout << "Row: " << row << '\n';

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
                    ShapiroWilkW(array, w, pw);

                    // cout << w << '\t' << pw << '\n';
                    Scalar mean, stdev;
                    cv::meanStdDev(normalized, mean, stdev);

                    double stderror = stdev[0] / normalized.rows; // Neighborhoods are square, thus sqrt(n) observations is number of rows

                    int c = 0;
                    if (stderror > mean[0] && mean[0] > -stderror &&
                        (1 + stderror > stdev[0] && stdev[0] > 1 - stderror) &&
                        pw < alpha)
                    {
                        c = 1;
                    }
                    double weight = c * w;

                    Wmax = max(weight, Wmax);

                    sumWeights += weight;
                    avg += weight * paddedImage.at<double>(sRow, sCol);
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

int main()
{
    // Ensure that program runs sequentially
    cv::setNumThreads(1);

    const string filename = "../../images/peppers_color.tif";
    const double sigma = 0.10;
    const int searchRadius = 10;
    const int neighborRadius = 3;

    Mat inputImage = imread(filename);
    cvtColor(inputImage, inputImage, COLOR_RGB2GRAY);

    Mat floatImage;
    inputImage.convertTo(floatImage, CV_64FC1, 1 / 255.0);

    Mat noise = floatImage.clone();
    randn(noise, 0, sigma);

    Mat noisyImage = floatImage.clone();
    cv::add(floatImage, noise, noisyImage);

    cout << "Noisy image PSNR: " << compute_psnr(floatImage, noisyImage) << '\n';

    imshow("original image", inputImage);
    imshow("noisy image", noisyImage);

    const chrono::system_clock::time_point start = chrono::high_resolution_clock::now();

    Mat denoised;
    swnlm(noisyImage, denoised, searchRadius, neighborRadius);

    const chrono::system_clock::time_point end = chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end - start;

    cout << "Finished in " << elapsed_seconds.count() << " seconds\n";

    const double denoisedPSNR = compute_psnr(floatImage, denoised);
    cout << "Denoised image PSNR: " << denoisedPSNR << '\n';

    //denoised.convertTo(denoised, inputImage.type(), 255.0);

    cv::imshow("Denoised", denoised);

    waitKey();
}
#include "swnlm.hpp"
#include "utils.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>

using namespace cv;
using namespace std;

typedef void (*NLMFunction)(const Mat &, Mat &, const double, const int, const int);

typedef struct DenoiseAlgorithm
{
    const string name;
    NLMFunction denoiser;

    DenoiseAlgorithm(const string &name, NLMFunction denoiser) : name(name), denoiser(denoiser) {}
} DenoiseAlgorithm;

/**
 * Save a Floating point image with value range [0..1] to png
 */
void saveImage(const string outputFile, const Mat &image)
{
    assert(image.type() == CV_64FC1);
    Mat converted;
    image.convertTo(converted, CV_8UC1, 255.0);
    cv::imwrite(outputFile + ".png", converted);
}

void writeResult(ofstream &resultsFile, const string &image, const double &sigma, const string &algorithm, const double &psnr, const double &ssim, const double &execTime)
{
    const string sigmaStr = to_string((int)sigma);
    const string psnrStr = to_string(psnr);
    const string ssimStr = to_string(ssim);
    string execTimeStr;
    if (isnan(execTime))
    {
        execTimeStr = "n/a";
    }
    else
    {
        execTimeStr = to_string(execTime);
    }

    resultsFile << image << ',' << sigmaStr << ',' << algorithm << ',' << psnrStr << ',' << ssimStr << ',' << execTimeStr << '\n';
}

void test(const vector<string> &images, const vector<DenoiseAlgorithm> &algorithms, const vector<double> &sigmas, const string &outputDir)
{
    const int searchRadius = 10;
    const int neighborRadius = 3;

    ofstream resultsFile;
    resultsFile.open(outputDir + "results.csv");
    resultsFile << "image,sigma,algorithm,psnr,SSIM,Execution time (s)\n";

    for (auto imagePath : images)
    {
        Mat inputImage = cv::imread(imagePath);
        assert(!inputImage.empty());

        if (inputImage.channels() == 3)
        {
            cv::cvtColor(inputImage, inputImage, cv::COLOR_RGB2GRAY);
        }
        assert(inputImage.channels() == 1);
        inputImage.convertTo(inputImage, CV_64FC1, 1.0 / 255);

        string imageName = imagePath.substr(imagePath.find_last_of("/") + 1); // Get filename
        imageName = imageName.substr(0, imageName.find_last_of('.'));         // Strip extension
        const string outputImagePath = outputDir + imageName;

        saveImage(outputImagePath + "_original", inputImage);
        writeResult(resultsFile, imageName, 0, "original", INFINITY, 1, NAN);

        cout << "Evaluating " << imageName << "...\n";

        Mat noise = inputImage.clone();
        Mat noisyImage = inputImage.clone();
        for (const double &sigma255 : sigmas)
        {
            cout << "\tsigma= " << sigma255 << '\n';

            const double sigma = sigma255 / 255;
            randn(noise, 0, sigma);
            noisyImage = inputImage + noise;

            saveImage(outputImagePath + "_sigma=" + to_string(sigma255) + "_noisy", noisyImage);
            double psnr = computePSNR(inputImage, noisyImage);
            double ssim = computeSSIM(inputImage, noisyImage);
            writeResult(resultsFile, imageName, sigma255, "noisy", psnr, ssim, NAN);

            for (const auto &algorithm : algorithms)
            {
                cout << "\t\tAlgorithm:" << algorithm.name;

                Mat denoisedImage;

                chrono::system_clock::time_point begin = chrono::high_resolution_clock::now();

                algorithm.denoiser(noisyImage, denoisedImage, sigma, searchRadius, neighborRadius);

                chrono::system_clock::time_point end = chrono::high_resolution_clock::now();

                double execTime = chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0;

                saveImage(outputImagePath + "_sigma=" + to_string(sigma255) + "_denoiser=" + algorithm.name, denoisedImage);
                psnr = computePSNR(inputImage, denoisedImage);
                ssim = computeSSIM(inputImage, denoisedImage);
                writeResult(resultsFile, imageName, sigma255, algorithm.name, psnr, ssim, execTime);

                cout << "\tPSNR: " << psnr << "\tSSIM: " << ssim << "\texec Time:" << execTime << '\n';
            }
        }
    }

    resultsFile.close();
}

int main()
{
    const string imageDir("../../../images/");
    const string outputDir("../../../output/");

    // Add all images in images dir for testing
    vector<string> images;
    for (const auto &entry : filesystem::directory_iterator(imageDir))
    {
        images.push_back(entry.path());
    }

    // Add functions to test
    vector<DenoiseAlgorithm> algorithms = {
        DenoiseAlgorithm("swnlm", &swnlm)};

    vector<double> sigmas = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60};

    test(images, algorithms, sigmas, outputDir);
}
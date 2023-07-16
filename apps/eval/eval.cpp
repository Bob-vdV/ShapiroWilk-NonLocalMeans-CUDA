#include "cnlm.hpp"
#include "cnlmcuda.cuh"
#include "swnlm.hpp"
#include "swnlmcuda.cuh"
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

template <typename Function>
struct DenoiseAlgorithm
{
    const string name;
    Function denoiser;

    DenoiseAlgorithm(const string &name, Function denoiser) : name(name), denoiser(denoiser) {}
};

/**
 * Save a Floating point image with value range [0..1] to png
 */
void saveImage(const string outputFile, const Mat &image)
{
    assert(image.type() == CV_64FC1);
    Mat converted;
    image.convertTo(converted, CV_8UC1);
    cv::imwrite(outputFile + ".png", converted);
}

void writeResult(ofstream &resultsFile, const string &image, const double &sigma, const int &searchRadius, const int &neighborRadius, const string &algorithm, const double &psnr, const double &ssim, const double &execTime)
{
    const string sigmaStr = to_string((int)sigma);
    const string psnrStr = to_string(psnr);
    const string ssimStr = to_string(ssim);
    string execTimeStr, searchRadiusStr, neighborRadiusStr;
    if (isnan(execTime))
    {
        execTimeStr = "-";
        searchRadiusStr = "-";
        neighborRadiusStr = "-";
    }
    else
    {
        execTimeStr = to_string(execTime);
        searchRadiusStr = to_string(searchRadius);
        neighborRadiusStr = to_string(neighborRadius);
    }

    resultsFile << image << ',' << sigmaStr << ',' << algorithm << ',' << searchRadiusStr << ',' << neighborRadiusStr << ',' << psnrStr << ',' << ssimStr << ',' << execTimeStr << '\n';
}

template <typename T>
void test(
    const vector<string> &images,
    const vector<DenoiseAlgorithm<NLMFunction<T>>> &algorithms,
    const vector<T> &sigmas,
    const vector<int> &searchRadii,
    const vector<int> &neighborRadii,
    const size_t repetitions,
    const string &outputDir)
{
    const double maxVal = 255;

    int totalResults = images.size() * algorithms.size() * sigmas.size() * searchRadii.size() * neighborRadii.size();
    int currentRes = 1;

    ofstream resultsFile;
    resultsFile.open(outputDir + "results.csv");
    resultsFile << "image,sigma,algorithm,search radius,neighbor radius,psnr,SSIM,execution time (s)\n";

    auto testStart = chrono::high_resolution_clock::now();
    for (auto imagePath : images)
    {
        Mat inputImage = cv::imread(imagePath);
        assert(!inputImage.empty());

        if (inputImage.channels() == 3)
        {
            cv::cvtColor(inputImage, inputImage, cv::COLOR_RGB2GRAY);
        }
        assert(inputImage.channels() == 1);
        inputImage.convertTo(inputImage, CV_64FC1);

        string imageName = imagePath.substr(imagePath.find_last_of("/") + 1); // Get filename
        imageName = imageName.substr(0, imageName.find_last_of('.'));         // Strip extension
        const string outputImagePath = outputDir + imageName;

        saveImage(outputImagePath + "_original", inputImage);
        writeResult(resultsFile, imageName, 0, 0, 0, "original", INFINITY, 1, NAN);

        cout << "Evaluating " << imageName << "...\n";

        Mat noise = inputImage.clone();
        Mat noisyImage = inputImage.clone();
        for (const T &sigma : sigmas)
        {
            cout << "\tsigma= " << sigma << '\n';

            randn(noise, 0, sigma);
            noisyImage = inputImage + noise;

            saveImage(outputImagePath + "_sigma=" + to_string(sigma) + "_noisy", noisyImage);
            double psnr = computePSNR<T>(inputImage, noisyImage, maxVal);
            double ssim = computeSSIM<T>(inputImage, noisyImage, maxVal);
            writeResult(resultsFile, imageName, sigma, 0, 0, "noisy", psnr, ssim, NAN);

            for (const int &searchRadius : searchRadii)
            {
                cout << "\t\tSearchRadius:" << searchRadius << '\n';

                for (const int &neighborRadius : neighborRadii)
                {
                    cout << "\t\tneighborRadius:" << neighborRadius << '\n';

                    for (const auto &algorithm : algorithms)
                    {
                        cout << "\t\t\tAlgorithm:" << algorithm.name << '\n';

                        double minTime = INFINITY;
                        Mat denoisedImage;

                        for (size_t rep = 0; rep < repetitions; rep++)
                        {

                            chrono::system_clock::time_point begin = chrono::high_resolution_clock::now();

                            algorithm.denoiser(noisyImage, denoisedImage, sigma, searchRadius, neighborRadius);

                            chrono::system_clock::time_point end = chrono::high_resolution_clock::now();

                            double execTime = chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0;
                            minTime = min(minTime, execTime);
                            cout << "\t\t\t\tRep " << rep << ": " << minTime << " seconds\n";
                        }

                        saveImage(outputImagePath + "_sigma=" + to_string(sigma) + "_searchRadius=" + to_string(searchRadius) + "_neighborRadius=" + to_string(neighborRadius) + "_denoiser=" + algorithm.name, denoisedImage);
                        psnr = computePSNR<T>(inputImage, denoisedImage, maxVal);
                        ssim = computeSSIM<T>(inputImage, denoisedImage, maxVal);
                        writeResult(resultsFile, imageName, sigma, searchRadius, neighborRadius, algorithm.name, psnr, ssim, minTime);

                        cout << "\tPSNR: " << psnr << "\tSSIM: " << ssim << "\texec Time:" << minTime << '\n';

                        auto currentTime = chrono::high_resolution_clock::now();
                        double testDuration = chrono::duration_cast<chrono::seconds>(currentTime - testStart).count();
                        int testsLeft = totalResults - currentRes;
                        double timePerResult = testDuration / currentRes;

                        cout << "Time left (s): " << testsLeft * timePerResult << '\n';
                        currentRes++;
                    }
                }
            }
        }
    }

    resultsFile.close();
}

template <typename T>
void runTests()
{
    // Ensure that program runs sequentially
    cv::setNumThreads(1);

    // Fix the seed to a constant
    cv::theRNG().state = 42;

    const string imageDir("../../../images/standard/");
    const string outputDir("../../../output/standard2/");

    // Create output directory
    filesystem::create_directory(outputDir);

    // Add all images in images dir for testing
    vector<string> images;
    for (const auto &entry : filesystem::directory_iterator(imageDir))
    {
        images.push_back(entry.path());
    }

    // Add functions to test
    vector<DenoiseAlgorithm<NLMFunction<T>>> algorithms = {
        DenoiseAlgorithm("swnlm", &swnlm<T>),
        DenoiseAlgorithm("cnlm", &cnlm<T>),
        DenoiseAlgorithm("swnlm_cuda", &swnlmcuda<T>),
        DenoiseAlgorithm("cnlm_cuda", &cnlmcuda<T>)};

    vector<T> sigmas = {10, 20, 30, 40, 50, 60};
    vector<int> searchRadii = {5, 8, 10};
    vector<int> neighborRadii = {1, 2, 3};

    const size_t repetitions = 3;

    test<T>(images, algorithms, sigmas, searchRadii, neighborRadii, repetitions, outputDir);
}

int main()
{
    runTests<uint8_t>();
}
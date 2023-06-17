#include "utils.hpp"
#include "swnlmcuda.cuh"

#include <iostream>

#include <filesystem> //TODO remove

using namespace std;

int main()
{
    const string filename = "/home/bob/Documents/Uni/BachelorThesis/NonLocalMeans/images/mandril.tif";// "../../../images/mandril.tif";
    const double sigma = 30.0 / 255;
    const int searchRadius = 10;
    const int neighborRadius = 3;

    testNLM(filename, sigma, searchRadius, neighborRadius, &swnlmcuda);
}
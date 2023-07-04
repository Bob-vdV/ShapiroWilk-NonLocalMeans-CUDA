#include "utils.hpp"
#include "cnlmcuda.cuh"

using namespace std;

int main()
{
    const string filename = "/home/bob/Documents/Uni/BachelorThesis/NonLocalMeans/images/mandril.tif";// "../../../images/mandril.tif";
    const double sigma = 30;
    const int searchRadius = 10;
    const int neighborRadius = 3;
    const bool showImg = true;

    testNLM(filename, sigma, searchRadius, neighborRadius, &cnlmcuda<double>, showImg);
}
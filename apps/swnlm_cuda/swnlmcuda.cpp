#include "utils.hpp"
#include "swnlmcuda.cuh"

using namespace std;

using NLMType = uint8_t;

int main()
{
    const string filename = "/home/bob/Documents/Uni/BachelorThesis/NonLocalMeans/images/standard/cameraman.png"; //"/home/bob/Documents/Uni/BachelorThesis/NonLocalMeans/images/mandril.tif";// "../../../images/mandril.tif";
    const NLMType sigma = 30;
    const int searchRadius = 10;
    const int neighborRadius = 3;//1;
    const bool showImg = true;

    testNLM<NLMType>(filename, sigma, searchRadius, neighborRadius, &swnlmcuda<NLMType>, showImg);
}
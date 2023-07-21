#include "utils.hpp"
#include "cnlmcuda.cuh"

using namespace std;

using NLMType = uint8_t;

int main()
{
    const string filename = "/home/bob/Documents/Uni/BachelorThesis/NonLocalMeans/images/standard/cameraman.png"; //"/home/bob/Documents/Uni/BachelorThesis/NonLocalMeans/images/polyu/book_1296x864.png";
    const NLMType sigma = 10;
    const int searchRadius = 10;
    const int neighborRadius = 3;
    const bool showImg = true;

    testNLM(filename, sigma, searchRadius, neighborRadius, &cnlmcuda<NLMType>, showImg);
}
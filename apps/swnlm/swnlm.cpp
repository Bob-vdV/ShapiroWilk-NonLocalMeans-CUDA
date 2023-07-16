#include "swnlm.hpp"
#include "utils.hpp"

#include <iostream>

using namespace std;

using NLMType = uint8_t;

int main()
{
    const string filename = "/home/bob/Documents/Uni/BachelorThesis/NonLocalMeans/images/standard/cameraman.png";
    const NLMType sigma = 30.0;
    const int searchRadius = 10;
    const int neighborRadius = 3;//1;//3;

    testNLM<NLMType>(filename, sigma, searchRadius, neighborRadius, &swnlm<NLMType>);
}
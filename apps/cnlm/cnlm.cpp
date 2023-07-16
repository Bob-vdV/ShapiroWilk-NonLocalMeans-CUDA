#include "cnlm.hpp"
#include "utils.hpp"

#include <iostream>

using namespace std;

using NLMType = double;

int main()
{
    const string filename = "/home/bob/Documents/Uni/BachelorThesis/NonLocalMeans/images/polyu/book_1296x864.png"; // "/home/bob/Documents/Uni/BachelorThesis/NonLocalMeans/images/standard/cameraman.png";
    const NLMType sigma = 30;
    const int searchRadius = 10;
    const int neighborRadius = 3;

    testNLM<NLMType>(filename, sigma, searchRadius, neighborRadius, &cnlm<NLMType>);
}
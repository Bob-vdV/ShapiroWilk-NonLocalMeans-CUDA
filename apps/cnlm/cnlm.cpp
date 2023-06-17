#include "cnlm.hpp"
#include "utils.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    const string filename = "../../../images/mandril.tif";
    const double sigma = 30.0 / 255;
    const int searchRadius = 10;
    const int neighborRadius = 3;

    testNLM(filename, sigma, searchRadius, neighborRadius, &cnlm);
}
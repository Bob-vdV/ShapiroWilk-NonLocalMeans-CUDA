#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/core.hpp>

typedef void (*NLMFunction)(const cv::Mat &, cv::Mat &, const double, const int, const int);

double computePSNR(const cv::Mat &baseImage, const cv::Mat &changedImage);

double computeSSIM(const cv::Mat &baseImage, const cv::Mat &changedImage);

void testNLM(const std::string filename, const double sigma, const int searchRadius, const int neighborRadius, NLMFunction nlmFunction);

#endif
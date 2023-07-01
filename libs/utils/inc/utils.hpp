#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/core.hpp>

typedef void (*NLMFunction_short)(const cv::Mat &, cv::Mat &, const short, const int, const int);
typedef void (*NLMFunction_float)(const cv::Mat &, cv::Mat &, const float, const int, const int);
typedef void (*NLMFunction_double)(const cv::Mat &, cv::Mat &, const double, const int, const int);

double computePSNR(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);

template<typename T>
double computeSSIM(const cv::Mat &baseImage, const cv::Mat &changedImage, const T max);

template <typename T, typename Function>
void testNLM(const std::string filename, const T sigma, const int searchRadius, const int neighborRadius, Function nlmFunction, const bool showImg = true);

#endif
#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/core.hpp>

double computePSNR(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);

template <typename T>
double computeSSIM(const cv::Mat &baseImage, const cv::Mat &changedImage, const double max);

template <typename T, typename Function>
void testNLM(const std::string filename, const T sigma, const int searchRadius, const int neighborRadius, Function nlmFunction, const bool showImg = true);

#endif
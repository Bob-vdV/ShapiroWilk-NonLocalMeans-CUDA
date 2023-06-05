#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/core.hpp>

double computePSNR(const cv::Mat &baseImage, const cv::Mat &changedImage);

double computeSSIM(const cv::Mat &baseImage, const cv::Mat &changedImage);

#endif
#ifndef SWNLM_HPP
#define SWNLM_HPP

#include <opencv2/core.hpp>

void swnlm(const cv::Mat &noisyImage, cv::Mat &denoised, const double sigma, const int searchRadius, const int neighborRadius);

#endif
#ifndef SWNLM_HPP
#define SWNLM_HPP

#include <opencv2/core.hpp>

template <typename T>
void swnlm(const cv::Mat &noisyImage, cv::Mat &denoised, const T sigma, const int searchRadius, const int neighborRadius);

#endif
#ifndef CNLM_HPP
#define CNLM_HPP

#include <opencv2/core.hpp>

template <typename T>
void cnlm(const cv::Mat &noisyImage, cv::Mat &denoised, const T sigma, const int searchRadius, const int neighborRadius);

#endif
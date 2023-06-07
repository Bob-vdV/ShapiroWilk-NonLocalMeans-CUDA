#ifndef CNLM_HPP
#define CNLM_HPP

#include <opencv2/core.hpp>

void cnlm(const cv::Mat &noisyImage, cv::Mat &denoised, const double sigma, const int searchRadius, const int neighborRadius);

#endif
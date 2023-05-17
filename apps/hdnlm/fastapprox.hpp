#ifndef NLM_FASTAPPROX_HPP
#define NLM_FASTAPPROX_HPP

#include <opencv2/core.hpp>

void fastApprox(const cv::Mat &inputImage, const int S, const double h, cv::Mat &center, const cv::Mat &guideImage, cv::Mat &outputImage);

#endif
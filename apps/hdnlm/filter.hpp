#ifndef NLM_FILTER_HPP
#define NLM_FILTER_HPP

#include <opencv2/core.hpp>

void boxFilter(const cv::Mat &inputMat, const int S, cv::Mat &outputMat);

#endif
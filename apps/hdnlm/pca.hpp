#ifndef NLM_PCA_HPP
#define NLM_PCA_HPP

#include <opencv2/core.hpp>

void computePca(const cv::Mat &inputMat, cv::Mat &outputMat, const int windowRadius, const int numDims);

#endif
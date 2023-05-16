#ifndef NLM_KMEANS_HPP
#define NLM_KMEANS_HPP

#include <opencv2/core.hpp>

void kmeansRecursive(const cv::Mat &inputMat, cv::Mat &outputMat, int clusters);

#endif
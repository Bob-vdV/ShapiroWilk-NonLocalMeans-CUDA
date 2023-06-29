#ifndef SWNLM_CUDA_CUH
#define SWNLM_CUDA_CUH

#include <opencv2/core.hpp>

void swnlmcuda(const cv::Mat &noisyImage, cv::Mat &denoised, const double sigma, const int searchRadius, const int neighborRadius);

#endif
#ifndef SWNLM_CUDA_CUH
#define SWNLM_CUDA_CUH

#include <opencv2/core.hpp>

template <typename T>
void swnlmcuda(const cv::Mat &noisyImage, cv::Mat &denoised, const T sigma, const int searchRadius, const int neighborRadius);

#endif
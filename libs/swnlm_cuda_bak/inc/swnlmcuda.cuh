#ifndef SWNLM_CUDA_CUH
#define SWNLM_CUDA_CUH

#include <opencv2/core.hpp>

template <typename T, int searchRadius, int neighborRadius>
void swnlmcuda(const cv::Mat &noisyImage, cv::Mat &denoised, const T sigma);

#endif
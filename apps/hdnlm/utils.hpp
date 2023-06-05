#ifndef HDNLM_UTILS_HH
#define HDNLM_UTILS_HH

#include <opencv2/core.hpp>

double computePSNR(const cv::Mat &baseImage, const cv::Mat &changedImage);

void resize3D(cv::Mat &inputMat, cv::Mat &outputMat, const int rows, const int cols);

template <typename T>
void mat3toMat2(const cv::Mat &inputMat, cv::Mat &outputMat, const int dim, const int idx);

template <typename T>
void mat2toMat3(const cv::Mat &inputMat, cv::Mat &outputMat, const int dim, const int idx);

#include "utils.tpp"

#endif
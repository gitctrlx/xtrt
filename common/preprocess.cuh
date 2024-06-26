#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>

/**
 * @brief Enumeration representing different scaling methods for preprocessing.
 */
enum class ScaleMethod {
  Resize,     ///< Resize method
  LetterBox   ///< LetterBox method
};

void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();
void cuda_preprocess(uint8_t* src, int src_width, int src_height,
                     float* dst, int dst_width, int dst_height);
void cuda_batch_preprocess(std::vector<cv::Mat>& img_batch,
                           float* dst, int dst_width, int dst_height);

void process_input_gpu(cv::Mat& input_img, float* input_device_buffer, ScaleMethod method);

void process_input_cv_affine(cv::Mat& src, float* input_device_buffer, ScaleMethod method);

void process_input_cpu(cv::Mat& src, float* input_device_buffer, ScaleMethod method);

cv::Mat resize_image(cv::Mat &src, int new_width, int new_height);
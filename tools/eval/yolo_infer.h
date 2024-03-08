#ifndef YOLOINFER_H
#define YOLOINFER_H

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "logger.h"
#include "common.h"
#include "buffers.h"

#include "preprocess.cuh"
#include "postprocess.h"
#include "types.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <fstream>

class YOLOInfer {
public:
    YOLOInfer(const std::string &engine_file, int preprocess_mode, int nBatchSize, int nChannel, int nHeight, int nWidth);
    ~YOLOInfer();
    std::vector<Detection> infer(const std::string &input_image_path, ScaleMethod method, bool end2end);

private:
    std::string engine_file_;
    int preprocess_mode_;
    int nBatchSize_, nChannel_, nHeight_, nWidth_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<unsigned char> load_engine_file(const std::string &file_name);
    void initialize();
    void preprocess(cv::Mat &frame, samplesCommon::BufferManager &buffers, ScaleMethod method);
    std::vector<Detection> postprocess(cv::Mat &frame, const samplesCommon::BufferManager &buffers, ScaleMethod method, bool end2end,
        std::chrono::high_resolution_clock::time_point start, 
        std::chrono::high_resolution_clock::time_point end);
};

#endif // YOLOINFER_H
#include "yolo_infer.h"
#include "preprocess.cuh"
#include "common.h"

#include <fstream>
#include <iostream>
#include <cassert>
#include <chrono>
#include <config.h>

YOLOInfer::YOLOInfer(const std::string &engine_file, int preprocess_mode, int nBatchSize, int nChannel, int nHeight, int nWidth)
    : engine_file_(engine_file), preprocess_mode_(preprocess_mode), nBatchSize_(nBatchSize), nChannel_(nChannel), nHeight_(nHeight), nWidth_(nWidth)
{
    initialize();
}

YOLOInfer::~YOLOInfer() {
    /* 
     * Since smart pointers are used, manual resource deallocation is not required.
     */
}

void YOLOInfer::initialize() {
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
    auto plan = load_engine_file(engine_file_);
    cuda_preprocess_init(nWidth_ *nHeight_);
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    context_->setInputShape(engine_->getIOTensorName(0), nvinfer1::Dims4(nBatchSize_, nChannel_, nHeight_, nWidth_));
}

std::vector<unsigned char> YOLOInfer::load_engine_file(const std::string &file_name) {
    std::ifstream engine_file(file_name, std::ios::binary);
    assert(engine_file.is_open() && "Unable to load engine file.");
    engine_file.seekg(0, std::ios::end);
    size_t length = engine_file.tellg();
    std::vector<unsigned char> engine_data(length);
    engine_file.seekg(0, std::ios::beg);
    engine_file.read(reinterpret_cast<char*>(engine_data.data()), length);
    return engine_data;
}

std::vector<Detection> YOLOInfer::infer(const std::string &input_image_path, ScaleMethod method, bool end2end) {
    cv::Mat frame = cv::imread(input_image_path);
    // std::cout<<input_image_path<<std::endl;
    if (frame.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return {};
    }

    samplesCommon::BufferManager buffers(engine_);

    // SimpleProfiler profiler("MyInferenceProfiler");
    // context_->setProfiler(&profiler);

    preprocess(frame, buffers, method);
    auto start = std::chrono::high_resolution_clock::now();
    context_->executeV2(buffers.getDeviceBindings().data());

    // std::cout << profiler << std::endl;

    buffers.copyOutputToHost();
    auto end = std::chrono::high_resolution_clock::now();
    auto detections = postprocess(frame, buffers, method, end2end, start, end); // Need to pass imageId

    // cv::imwrite("img.jpg", frame);

    return detections;
}

inline void YOLOInfer::preprocess(cv::Mat &frame, samplesCommon::BufferManager &buffers, ScaleMethod method) {
    switch (preprocess_mode_) {
        case 0:
            // CPU preprocessing
            process_input_cpu(frame, static_cast<float*>(buffers.getDeviceBuffer(kInputTensorName)), method);
            break;
        case 1:
            // CPU + GPU preprocessing
            process_input_cv_affine(frame, static_cast<float*>(buffers.getDeviceBuffer(kInputTensorName)), method);
            break;
        case 2:
            // GPU preprocessing
            process_input_gpu(frame, static_cast<float*>(buffers.getDeviceBuffer(kInputTensorName)), method);
            break;
        default:
            std::cerr << "[E] Preprocessing method not selected!" << std::endl;
            break;
    }
}

inline std::vector<Detection> YOLOInfer::postprocess(cv::Mat &frame, const samplesCommon::BufferManager &buffers, ScaleMethod method, bool end2end, 
    std::chrono::high_resolution_clock::time_point start, 
    std::chrono::high_resolution_clock::time_point end) {

    int32_t *num_det = static_cast<int32_t*>(buffers.getHostBuffer(kOutNumDet));
    int32_t     *cls = static_cast<int32_t*>(buffers.getHostBuffer(kOutDetCls));
    float      *conf = static_cast<float*>(buffers.getHostBuffer(kOutDetScores));
    float      *bbox = static_cast<float*>(buffers.getHostBuffer(kOutDetBBoxes));

    // Perform NMS (Non-Maximum Suppression) to get the final detection boxes
    std::vector<Detection> detections;
    if (!end2end) yolo_nms(detections, num_det, cls, conf, bbox, kConfThresh, kNmsThresh);
    for (int i = 0; i < *num_det; ++i) {
        Detection det;
        cv::Rect2f r;
        if (method == ScaleMethod::LetterBox) {
            r = get_rect(frame, &bbox[i * 4]);
        } else {
            r = get_rect_resize(frame, &bbox[i * 4]);
        }
        det.bbox[0] = r.x;
        det.bbox[1] = r.y;
        det.bbox[2] = r.x + r.width;
        det.bbox[3] = r.y + r.height;

        det.conf = conf[i];
        det.class_id = cls[i];
        detections.push_back(det);
        
    }

    // // Adjust each detection's bbox to the original image size
    // for (size_t j = 0; j < detections.size(); j++) {
    //     cv::Rect r = get_rect(frame, detections[j].bbox);

    //     // Update the bbox in the Detection struct with the adjusted coordinates
    //     detections[j].bbox[0] = r.x;
    //     detections[j].bbox[1] = r.y;
    //     detections[j].bbox[2] = r.x + r.width;
    //     detections[j].bbox[3] = r.y + r.height;

    //     // Optional: Draw the adjusted bounding boxes and class IDs on the frame
    //     cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
    //     cv::putText(frame, std::to_string(static_cast<int>(detections[j].class_id)), cv::Point(r.x, r.y - 10), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x27, 0xC1, 0x36), 2);
    // }

    // // Calculate and display code execution time and frame rate, draw text on the image
    // auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f;
    // auto time_str = std::to_string(elapsed) + "ms";
    // auto fps = 1000.0f / elapsed;
    // auto fps_str = std::to_string(fps) + "fps";

    // cv::putText(frame, time_str, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    // cv::putText(frame, fps_str, cv::Point(50, 100), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);

    return detections;
}


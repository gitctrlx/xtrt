#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "logger.h"
#include "common.h"
#include "buffers.h"

#include "preprocess.cuh"
#include "postprocess.h"
#include "types.h"


inline std::vector<unsigned char> load_engine_file(const std::string &file_name);

int main(int argc, char **argv)
{
    /*
     * Execute inference parameters
     */

    if (argc != 9)
    {
        std::cerr << "[Usage]: " << argv[0] << "\n"
                << "File Settings:\n"
                << "    [engine_file]        : Path to the engine file\n"
                << "    [input_image_path]   : Path to the input image file\n"
                << "    [output_image_path]  : Path to the output image file\n"

                << "Preprocess modee:\n"
                << "    [preprocess mode]    : Pre processing mode (cpu(0), cpu+gpu(1), gpu(2))\n\n"

                << "Dynamic shape input shape:\n"
                << "    [batch_size]         : Batch size for input batchsize (e.g., 1)\n"
                << "    [channels]           : Number of channels (e.g., 3 for RGB)\n"
                << "    [height]             : Height of the input images (e.g., 640)\n"
                << "    [width]              : Width of the input images (e.g., 640)" 
                << std::endl;
        return -1;
    }

    auto       engine_file = argv[1];
    auto  input_image_path = argv[2];
    auto output_image_path = argv[3];

    auto   preprocess_mode = std::stoi(argv[4]);

    int         nBatchSize = std::stoi(argv[5]);
    int           nChannel = std::stoi(argv[6]);
    int            nHeight = std::stoi(argv[7]);
    int             nWidth = std::stoi(argv[8]);


    /*
     * Initialize
     */

    cv::Mat          frame = cv::imread(input_image_path);;
    auto               cap = cv::VideoCapture(input_image_path);
    int              width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int             height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    auto              plan = load_engine_file(engine_file);

    cuda_preprocess_init(width * height); // Apply for CUDA memory
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");


    /*
     * Initialize: runtime, engine, context
     */

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    auto  engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    context->setInputShape(engine->getIOTensorName(0), nvinfer1::Dims32 {4, {nBatchSize, nChannel, nHeight, nWidth}});


    /*
    * Recurrent inference processing
    */

    // buffer: Manage buffer
    samplesCommon::BufferManager buffers(engine);

    // Statistical runtime
    auto start = std::chrono::high_resolution_clock::now();

    // Select preprocessing method
    switch (preprocess_mode) {
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

    /*
    * Execute inference
    */
    context->executeV2(buffers.getDeviceBindings().data());

    // Copy inference results back to host
    buffers.copyOutputToHost();

    // Get inference output from buffer manager
    int32_t *num_det = (int32_t *)buffers.getHostBuffer(kOutNumDet);     // Number of detected objects
    int32_t     *cls = (int32_t *)buffers.getHostBuffer(kOutDetCls);     // Detected object categories
    float      *conf = (float   *)buffers.getHostBuffer(kOutDetScores);  // Confidence levels of detected objects
    float      *bbox = (float   *)buffers.getHostBuffer(kOutDetBBoxes);  // Bounding boxes of detected objects
    

    /*
    * Post processing
    * Because smart pointers are used, there is no need to manually release resources
    */

    // Execute NMS (non maximum suppression) to obtain the final detection box
    std::vector<Detection> bboxs;
    yolo_nms(bboxs, num_det, cls, conf, bbox, kConfThresh, kNmsThresh);

    // Calculate and display code execution time (in milliseconds) and frame rate (FPS)
    auto end = std::chrono::high_resolution_clock::now();
    
    auto  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f; //1 microsecond = 0.001 milliseconds = 0.000001 seconds
    auto time_str = std::to_string(elapsed) + "ms";
    auto      fps = 1000.0f / elapsed;
    auto  fps_str = std::to_string(fps) + "fps";

    // Traverse the inference detection results, Draw bounding boxes and text on images
    for (size_t j = 0; j < bboxs.size(); j++){
        cv::Rect2f r = get_rect(frame, bboxs[j].bbox);
        cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(frame, std::to_string((int)bboxs[j].class_id), cv::Point(r.x, r.y - 10), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x27, 0xC1, 0x36), 2);
    }
    cv::putText(frame, time_str, cv::Point(50,  50), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    cv::putText(frame,  fps_str, cv::Point(50, 100), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);

    // OpenCV: Display
    // cv::imshow("frame", frame);

    // OpenCV: writing video files
    cv::imwrite(output_image_path, frame);
    std::cout << "[I] Image processed." << std::endl;
}

inline std::vector<unsigned char> load_engine_file(const std::string &file_name)
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(file_name, std::ios::binary);
    assert(engine_file.is_open() && "Unable to load engine file.");
    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);
    return engine_data;
}
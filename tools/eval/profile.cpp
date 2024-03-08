#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "logger.h"
#include "common.h"
#include "buffers.h"

#include "preprocess.cuh"
#include "postprocess.h"
#include "types.h"
#include "config.h"

inline std::vector<unsigned char> load_engine_file(const std::string &file_name);

int main(int argc, char **argv)
{
    /*
     * Execute inference parameters
     */

    if (argc != 8)
    {
        std::cerr << "[Usage]: " << argv[0] << "\n"
                << "File Settings:\n"
                << "    [engine_file]        : Path to the engine file\n"
                << "    [input_image_path]   : Path to the input image file\n"

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

    auto   preprocess_mode = std::stoi(argv[3]);

    int         nBatchSize = std::stoi(argv[4]);
    int           nChannel = std::stoi(argv[5]);
    int            nHeight = std::stoi(argv[6]);
    int             nWidth = std::stoi(argv[7]);


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

    samplesCommon::BufferManager buffers(engine);

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

    SimpleProfiler profiler("InferenceProfiler");
    context->setProfiler(&profiler);
    context->executeV2(buffers.getDeviceBindings().data());
    std::cout << profiler << std::endl;

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
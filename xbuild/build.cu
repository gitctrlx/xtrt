#include <cassert>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "calibrator.h"

#include "logger.h"


int main(int argc, char **argv)
{

    /*
     * Engine Build Parameters
     */

    if (argc != 24){
        std::cerr << "[Usage]: " << argv[0] << "\n"
                << "File and Model Settings:\n"
                << "    [onnx_file]          : Path to ONNX file\n"
                << "    [trt_file]           : Path to TRT file\n"
                << "    [accuracy]           : Accuracy setting (e.g., 'fp16')\n"
                << "    [optimization_level] : Optimization level (e.g., 3)\n\n"

                << "Dynamic Shape Settings:\n"
                << "    [min batch_size]        : Batch size 1 (e.g., 1)\n"
                << "    [opt batch_size]        : Batch size 2 (e.g., 4)\n"
                << "    [max batch_size]        : Batch size 3 (e.g., 8)\n"
                << "    [min channel]           : Channel 1 (e.g., 3)\n"
                << "    [opt channel]           : Channel 2 (e.g., 3)\n"
                << "    [max channel]           : Channel 3 (e.g., 3)\n"
                << "    [min height]            : Height 1 (e.g., 640)\n"
                << "    [opt height]            : Height 2 (e.g., 640)\n"
                << "    [max height]            : Height 3 (e.g., 640)\n"
                << "    [min width]             : Width 1 (e.g., 640)\n"
                << "    [opt width]             : Width 2 (e.g., 640)\n"
                << "    [max width]             : Width 3 (e.g., 640)\n"

                << "Calibration and Cache File Settings:\n"
                << "    [calibration]        : Calibration count (e.g., 120)\n\n"
                << "    [calibdata_path]     : Path to calibration data\n"
                << "    [calibdata_list]     : Calibration data list\n"
                << "    [int8cache_file]     : Path to INT8 cache file\n\n"

                << "Runtime and Configuration Settings:\n"
                << "    [use_time_cache]     : Use time cache ('true' or 'false')\n"
                << "    [ignore_mismatch]    : Ignore mismatch ('true' or 'false')\n"
                << "    [timing_cache_file]  : Path to timing cache file" 
                << std::endl;
        return -1;
    }

    std::string        onnxFile = argv[1];
    std::string         trtFile = argv[2];
    std::string        accuracy = argv[3];
    int      optimization_level = std::stoi(argv[4]);

    std::vector<int> nBatchSize = {std::stoi(argv[5]), std::stoi(argv[6]), std::stoi(argv[7])};
    std::vector<int>   nChannel = {std::stoi(argv[8]), std::stoi(argv[9]), std::stoi(argv[10])};
    std::vector<int>    nHeight = {std::stoi(argv[11]), std::stoi(argv[12]), std::stoi(argv[13])};
    std::vector<int>     nWidth = {std::stoi(argv[14]), std::stoi(argv[15]), std::stoi(argv[16])};

    int            nCalibration = std::stoi(argv[17]);
    std::string   calibdataPath = argv[18];
    std::string   calibdataList = argv[19];
    std::string   int8cacheFile = argv[20];

    bool          bUseTimeCache = (std::string(argv[21]) == "true");
    bool        bIgnoreMismatch = (std::string(argv[22]) == "true");
    std::string timingCacheFile = argv[23];


    /*
     * Initialize: Builder, Network, OptimizationProfile, uilderConfig
     */

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(xtrt::gLogger.getTRTLogger()));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    auto profile = builder->createOptimizationProfile();
    auto  config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());


    /*
     * ONNXx Parser: Using the onnxparser parser to parse the onnx model
     */

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, xtrt::gLogger.getTRTLogger()));
    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(xtrt::gLogger.getReportableSeverity()))){
        std::cout << std::string("[E] Failed parsing .onnx file!") << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i){
            auto *error = parser->getError(i);
            std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
        }
        return -1;
    }


    /*
     * Builder Config
     */

    // Dynamic Shape
    nvinfer1::ITensor *inputTensor = network->getInput(0);
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{nBatchSize[0], nChannel[0], nHeight[0], nWidth[0]});
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{nBatchSize[1], nChannel[1], nHeight[1], nWidth[1]});
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{nBatchSize[2], nChannel[2], nHeight[2], nWidth[2]});
    config->addOptimizationProfile(profile);

    // Config: Profile CUDA Stream
    auto profileStream = xtrtCommon::makeCudaStream();
    if (!profileStream){
        std::cout<<"[E] Failed making profile cuda stream!"<<std::endl;
        return -1;
    }
    config->setProfileStream(*profileStream);

    // Mixed accuracy: pf32, fp16, int8
    nvinfer1::IInt8Calibrator  *calibrator   = nullptr;
    if (accuracy == "fp16") {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } 
    else if (accuracy == "int8" && builder->platformHasFastInt8()) {
        Calibrator* calibrator = new Calibrator(
                                            nCalibration,calibdataPath, int8cacheFile, calibdataList, 
                                            nvinfer1::Dims4{nBatchSize[1], nChannel[1], nHeight[1], nWidth[1]}, 
                                            inputTensor->getName(),
                                            1);

        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator);
        config->setCalibrationProfile(profile);
    }

    // Config: Builder Optimization Level
    config->setBuilderOptimizationLevel(optimization_level);

    // Config(Debug): NVTX Tracing
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    // Config: Timing Cache
    std::vector<char> timingCacheData;
    if (bUseTimeCache && std::ifstream(timingCacheFile, std::ios::binary).good()) {
        std::ifstream cacheStream(timingCacheFile, std::ios::binary);
        cacheStream.seekg(0, std::ios::end);
        size_t size = cacheStream.tellg();
        cacheStream.seekg(0, std::ios::beg);
        timingCacheData.resize(size);
        cacheStream.read(timingCacheData.data(), size);
        std::cout << "[I] Succeeded loading " << timingCacheFile << std::endl;
    } else {
        std::cout << "[I] Failed loading " << timingCacheFile << " or using new cache" << std::endl;
    }

    std::unique_ptr<nvinfer1::ITimingCache> timingCache;
    if (bUseTimeCache) {
        timingCache.reset(config->createTimingCache(timingCacheData.data(), timingCacheData.size()));
        config->setTimingCache(*timingCache, bIgnoreMismatch);
    }


    /*
     * Serialization engine, deserialization engine testing, save engine
     */

    // Serialization engine
    auto engineString = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (engineString == nullptr || engineString->size() == 0){
        std::cout << "[E] Failed building serialized engine!" << std::endl;
        return 1;
    }
    std::cout << "[I] Succeeded building serialized engine!" << std::endl;

    // Deserialization engine testing
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(xtrt::gLogger.getTRTLogger()));
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineString->data(), engineString->size()));
    if (engine == nullptr){
        std::cout << "[E] Failed building engine!" << std::endl;
        return 1;
    }

    // Save engine
    std::ofstream engineFile(trtFile, std::ios::binary);
    if (!engineFile){
        std::cout << "[E] Failed opening file to write" << std::endl;
        return 1;
    }
    engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
    if (engineFile.fail()){
        std::cout << "[E] Failed saving .plan file!" << std::endl;
        return 1;
    }
    std::cout << "[I] Succeeded saving .plan file!" << std::endl;


    /*
     * Save timing Cache
     */

    if (bUseTimeCache) {
        nvinfer1::IHostMemory* serializedCache = timingCache->serialize();
        std::ofstream outputCacheStream(timingCacheFile, std::ios::binary);
        outputCacheStream.write(static_cast<const char*>(serializedCache->data()), serializedCache->size());
        std::cout << "[I] Succeeded saving " << timingCacheFile << std::endl;
        serializedCache->destroy();
    }


    /*
     * Release resources
     * Because smart pointers are used, there is no need to manually release resources
     */

    if (accuracy == "int8" && calibrator != nullptr){
        delete calibrator;
    }


    std::cout << "[I] Engine build success!" << std::endl;
    return 0;
}
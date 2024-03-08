#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "common.h"
#include "preprocess.cuh"

class Calibrator : public nvinfer1::IInt8MinMaxCalibrator
{
private:
    int                      mCalibration;
    std::string              mcalibrationDataFile;
    std::string              mint8CacheFile;
    nvinfer1::Dims4          mInputDims;
    const char              *mInputTensorName;
    int                      mBatchSize;

    int                      mImgSize;
    int                      mInputCount;
    std::vector<std::string> mFileNames;
    int                      mCurBatch{0};
    float                   *mDeviceBatchData{nullptr};
    std::vector<char>        mCalibrationCache;
    
public:
    Calibrator(const int nCalibration, const std::string &calibrationDataFile, const std::string &int8CacheFile, const std::string &calibrationDataList, const nvinfer1::Dims4 inputShape, const char* mInputTensorName, int batchSize);
    
    int32_t     getBatchSize() const noexcept;
    bool        getBatch(void *bindings[], char const *names[], int32_t nbBindings) noexcept;
    void const *readCalibrationCache(std::size_t &length) noexcept;
    void        writeCalibrationCache(void const *ptr, std::size_t length) noexcept;
};

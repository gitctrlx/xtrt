#include "calibrator.h"
#include "config.h"

Calibrator::Calibrator(const int nCalibration, const std::string &calibrationDataFile, const std::string &int8CacheFile, const std::string &calibrationDataList, const nvinfer1::Dims4 inputShape, const char* inputTensorName, int batchSize):
   mCalibration(nCalibration), mcalibrationDataFile(calibrationDataFile), mint8CacheFile(int8CacheFile),  mInputDims(inputShape), mInputTensorName(inputTensorName), mBatchSize(batchSize)
{
    mImgSize = mInputDims.d[2] * mInputDims.d[3];
    mInputCount = mBatchSize * samplesCommon::volume(mInputDims);
    cuda_preprocess_init(mImgSize);
    cudaMalloc(&mDeviceBatchData, mImgSize * 3 * sizeof(float));

    std::ifstream infile(calibrationDataList);
    std::string line;
    while (std::getline(infile, line)){
        sample::gLogInfo << line << std::endl;
        mFileNames.push_back(line);
    }

    // mCalibration = mFileNames.size() / mBatchSize;
    std::cout << "[I]CalibrationDataReader: " << mFileNames.size() << " images, " << nCalibration << " batches." << std::endl;
}

int32_t Calibrator::getBatchSize() const noexcept
{
    return mBatchSize; // Dynamic shape:1
}

bool Calibrator::getBatch(void *bindings[], char const *names[], int32_t nbBindings) noexcept
{
    if (mCurBatch >= mCalibration){
        return false;
    }

    int offset = mImgSize * 3 * sizeof(float);
    for (int i = 0; i < mBatchSize; i++){
        int idx = mCurBatch * mBatchSize + i;
        std::string fileName = mcalibrationDataFile + "/" + mFileNames[idx];
        cv::Mat img = cv::imread(fileName);
        int new_img_size = img.cols * img.rows;
        if (new_img_size > mImgSize){
            mImgSize = new_img_size;
            cuda_preprocess_destroy();
            cuda_preprocess_init(mImgSize);
        }
        process_input_gpu(img, mDeviceBatchData + i * offset, method);
    }
    
    for (int i = 0; i < nbBindings; i++){
        if (!strcmp(names[i], mInputTensorName)){
            bindings[i] = mDeviceBatchData + i * offset;
        }
    }

    mCurBatch++;
    return true;
}

void const *Calibrator::readCalibrationCache(std::size_t &length) noexcept
{
    mCalibrationCache.clear();
    std::ifstream input(mint8CacheFile, std::ios::binary);
    input >> std::noskipws;
    if (input.good()){
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                  std::back_inserter(mCalibrationCache));
    }
    length = mCalibrationCache.size();
    return length ? mCalibrationCache.data() : nullptr;
}

void Calibrator::writeCalibrationCache(void const *cache, std::size_t length) noexcept
{
    std::ofstream output(mint8CacheFile, std::ios::binary);
    output.write(reinterpret_cast<const char *>(cache), length);
}

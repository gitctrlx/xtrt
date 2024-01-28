#pragma once

/* --------------------------------------------------------
 * These configs are related to tensorrt model, if these are changed,
 * please re-compile and re-serialize the tensorrt model.
 * --------------------------------------------------------*/

// These are used to define input/output tensor names,
// you can set them to whatever you want.
const static char* kInputTensorName = "images";
const static char* kOutputTensorName = "prob";

// const static char* kOutNumDet = "DecodeNumDetection";
// const static char* kOutDetScores = "DecodeDetectionScores";
// const static char* kOutDetBBoxes = "DecodeDetectionBoxes";
// const static char* kOutDetCls = "DecodeDetectionClasses";

const static char* kOutNumDet = "num_dets";
const static char* kOutDetScores = "scores";
const static char* kOutDetBBoxes = "boxes";
const static char* kOutDetCls = "labels";

// Detection model and Segmentation model' number of classes
constexpr static int kNumClass = 80;

// Classfication model's number of classes
constexpr static int kClsNumClass = 1000;

constexpr static int kBatchSize = 1;

// Yolo's input width and height must by divisible by 32
constexpr static int kInputH = 640;
constexpr static int kInputW = 640;

// Classfication model's input shape
constexpr static int kClsInputH = 224;
constexpr static int kClsInputW = 224;

// Maximum number of output bounding boxes from yololayer plugin.
// That is maximum number of output bounding boxes before NMS.
constexpr static int kMaxNumOutputBbox = 1000;

constexpr static int kNumAnchor = 3;

// The bboxes whose confidence is lower than kIgnoreThresh will be ignored in yololayer plugin.
constexpr static float kIgnoreThresh = 0.1f;

/* --------------------------------------------------------
 * These configs are NOT related to tensorrt model, if these are changed,
 * please re-compile, but no need to re-serialize the tensorrt model.
 * --------------------------------------------------------*/

// NMS overlapping thresh and final detection confidence thresh
const static float kNmsThresh = 0.7f;
const static float kConfThresh = 0.4f;

const static int kGpuId = 0;

// If your image size is larger than 4096 * 3112, please increase this value
const static int kMaxInputImageSize = 4096 * 3112;


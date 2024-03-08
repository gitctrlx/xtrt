#include "postprocess.h"
#include "utils.h"


/**
 * @brief Clamps a value between a minimum and maximum value.
 * 
 * @param val The value to be clamped.
 * @param minVal The minimum value.
 * @param maxVal The maximum value.
 * @return The clamped value.
 */
float clamp(const float val, const float minVal, const float maxVal)
{
  assert(minVal <= maxVal);
  return std::min(maxVal, std::max(minVal, val));
}


/**
 * @brief Calculates and returns a cv::Rect object based on the input image and bounding box coordinates.
 * 
 * @param img The input image.
 * @param bbox An array of four floating-point values representing the bounding box coordinates [x1, y1, x2, y2].
 * @return cv::Rect The calculated cv::Rect object.
 */
cv::Rect2f get_rect(cv::Mat &img, float bbox[4])
{
  float scale = std::min(kInputH / float(img.rows), kInputW / float(img.cols));
  int offsetx = (kInputW - int(img.cols * scale)) / 2;
  int offsety = (kInputH - int(img.rows * scale)) / 2;

  float output_width = img.cols;
  float output_height = img.rows;

  float x1 = (bbox[0] - offsetx) / scale;
  float y1 = (bbox[1] - offsety) / scale;
  float x2 = (bbox[2] - offsetx) / scale;
  float y2 = (bbox[3] - offsety) / scale;

  x1 = std::max(x1, 0.0f); 
  y1 = std::max(y1, 0.0f);
  x2 = std::min(x2, output_width);
  y2 = std::min(y2, output_height);

  float left = x1;
  float top = y1;
  float width = std::max(x2 - x1, 0.0f);
  float height = std::max(y2 - y1, 0.0f);

  return cv::Rect2f(left, top, width, height);
}

/**
 * @brief Calculates and returns a resized rectangle based on the input image and bounding box coordinates.
 * 
 * @param img The input image.
 * @param bbox An array of four floating-point values representing the bounding box coordinates [x1, y1, x2, y2].
 * @return cv::Rect2f The resized rectangle.
 */
cv::Rect2f get_rect_resize(cv::Mat &img, float bbox[4]) {
  float scaleX = img.cols / float(kInputW);
  float scaleY = img.rows / float(kInputH);

  float x1 = bbox[0] * scaleX;
  float y1 = bbox[1] * scaleY;
  float x2 = bbox[2] * scaleX;
  float y2 = bbox[3] * scaleY;

  x1 = std::max(x1, 0.0f);
  y1 = std::max(y1, 0.0f);
  x2 = std::min(x2, float(img.cols));
  y2 = std::min(y2, float(img.rows));

  float width = std::max(x2 - x1, 0.0f);
  float height = std::max(y2 - y1, 0.0f);

  return cv::Rect2f(x1, y1, width, height);
}

/**
 * Calculates the Intersection over Union (IoU) between two bounding boxes.
 * 
 * @param lbox The coordinates of the left bounding box [x, y, width, height].
 * @param rbox The coordinates of the right bounding box [x, y, width, height].
 * @return The IoU value between the two bounding boxes.
 */
float iou(float lbox[4], float rbox[4])
{
  float interBox[] = {
      (std::max)(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), // left
      (std::min)(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), // right
      (std::max)(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), // top
      (std::min)(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), // bottom
  };

  if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
    return 0.0f;

  float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
  return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}


/**
 * @brief Performs non-maximum suppression (NMS) on a vector of detections.
 * 
 * @param res The vector to store the filtered detections.
 * @param output The output array containing the detections.
 * @param conf_thresh The confidence threshold for filtering detections.
 * @param nms_thresh The IoU threshold for suppressing overlapping detections.
 */
void nms(std::vector<Detection> &res, float *output, float conf_thresh, float nms_thresh)
{
  int det_size = sizeof(Detection) / sizeof(float);
  std::map<float, std::vector<Detection>> m;
  for (int i = 0; i < output[0] && i < kMaxNumOutputBbox; i++)
  {
    if (output[1 + det_size * i + 4] <= conf_thresh)
      continue;
    Detection det;
    memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
    if (m.count(det.class_id) == 0)
      m.emplace(det.class_id, std::vector<Detection>());
    m[det.class_id].push_back(det);
  }
  for (auto it = m.begin(); it != m.end(); it++)
  {
    auto &dets = it->second;
    std::sort(dets.begin(), dets.end(), cmp);
    for (size_t m = 0; m < dets.size(); ++m)
    {
      auto &item = dets[m];
      res.push_back(item);
      for (size_t n = m + 1; n < dets.size(); ++n)
      {
        if (iou(item.bbox, dets[n].bbox) > nms_thresh)
        {
          dets.erase(dets.begin() + n);
          --n;
        }
      }
    }
  }
}


/**
 * Applies batched non-maximum suppression (NMS) to the given detection results.
 * 
 * @param res_batch The vector of detection results for each batch.
 * @param output The pointer to the output array containing the detection results.
 * @param batch_size The number of batches.
 * @param output_size The size of the output array.
 * @param conf_thresh The confidence threshold for filtering detections.
 * @param nms_thresh The overlap threshold for suppressing overlapping detections.
 */
void batch_nms(std::vector<std::vector<Detection>> &res_batch, float *output, int batch_size, int output_size, float conf_thresh, float nms_thresh)
{
  res_batch.resize(batch_size);
  for (int i = 0; i < batch_size; i++)
  {
    nms(res_batch[i], &output[i * output_size], conf_thresh, nms_thresh);
  }
}


/**
 * Draws bounding boxes on a batch of images.
 * 
 * @param img_batch The batch of input images.
 * @param res_batch The batch of detection results.
 */
void draw_bbox(std::vector<cv::Mat> &img_batch, std::vector<std::vector<Detection>> &res_batch)
{
  for (size_t i = 0; i < img_batch.size(); i++)
  {
    auto &res = res_batch[i];
    cv::Mat img = img_batch[i];
    for (size_t j = 0; j < res.size(); j++)
    {
      cv::Rect2f r = get_rect(img, res[j].bbox);
      cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
      cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
  }
}


/**
 * @brief Calculates and returns a downscaled rectangle based on the given bounding box and scale.
 * 
 * @param bbox The bounding box coordinates [left, top, width, height].
 * @param scale The scale factor to downscale the rectangle.
 * @return The downscaled rectangle.
 */
// static cv::Rect get_downscale_rect(float bbox[4], float scale)
// {
//   float left = bbox[0] - bbox[2] / 2;
//   float top = bbox[1] - bbox[3] / 2;
//   float right = bbox[0] + bbox[2] / 2;
//   float bottom = bbox[1] + bbox[3] / 2;
//   left /= scale;
//   top /= scale;
//   right /= scale;
//   bottom /= scale;
//   return cv::Rect(round(left), round(top), round(right - left), round(bottom - top));
// }


/**
 * Applies non-maximum suppression (NMS) to a list of detections based on confidence scores and bounding boxes.
 * 
 * @param res The vector to store the filtered detections.
 * @param num_det Pointer to the number of detections.
 * @param cls Pointer to the class IDs of the detections.
 * @param conf Pointer to the confidence scores of the detections.
 * @param bbox Pointer to the bounding boxes of the detections.
 * @param conf_thresh The confidence threshold for filtering detections.
 * @param nms_thresh The IoU (Intersection over Union) threshold for NMS.
 */
void yolo_nms(std::vector<Detection> &res, int32_t *num_det, int32_t *cls, float *conf, float *bbox, float conf_thresh, float nms_thresh)
{
  std::map<float, std::vector<Detection>> m;
  for (int i = 0; i < num_det[0]; i++)
  {
    if (conf[i] <= conf_thresh)
      continue;
    Detection det;
    det.bbox[0] = bbox[i * 4 + 0];
    det.bbox[1] = bbox[i * 4 + 1];
    det.bbox[2] = bbox[i * 4 + 2];
    det.bbox[3] = bbox[i * 4 + 3];
    det.conf = conf[i];
    det.class_id = cls[i];
    if (m.count(det.class_id) == 0)
      m.emplace(det.class_id, std::vector<Detection>());
    m[det.class_id].push_back(det);
  }
  for (auto it = m.begin(); it != m.end(); it++)
  {
    auto &dets = it->second;
    std::sort(dets.begin(), dets.end(), cmp);
    for (size_t m = 0; m < dets.size(); ++m)
    {
      auto &item = dets[m];
      res.push_back(item);
      for (size_t n = m + 1; n < dets.size(); ++n)
      {
        if (iou(item.bbox, dets[n].bbox) > nms_thresh)
        {
          dets.erase(dets.begin() + n);
          --n;
        }
      }
    }
  }
}
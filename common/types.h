#pragma once

#include "config.h"

/**
 * @brief Structure representing the parameters of the YOLO kernel.
 */
struct YoloKernel {
  int width; /**< The width of the kernel. */
  int height; /**< The height of the kernel. */
  float anchors[kNumAnchor * 2]; /**< The anchor values for the kernel. */
};

/**
 * @brief Structure representing the detection results.
 */
struct Detection {
  float bbox[4]; /**< The bounding box coordinates (xmin, ymin, xmax, ymax). */
  float conf; /**< The confidence score of the detection. */
  float class_id; /**< The class ID of the detection. */
};

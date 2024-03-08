/**
 * @file preprocess.cu
 * @brief CUDA implementation of image preprocessing functions.
 */

#include "preprocess.cuh"
#include "utils.h"
#include "config.h"

/**
 * @brief This file contains the definition of the `AffineMatrix` struct and the `img_buffer_device` variable.
 */

static uint8_t *img_buffer_device = nullptr;


/**
 * @brief This struct represents an affine transformation matrix.
 * The matrix is stored as a 1D array of 6 float values.
 */
struct AffineMatrix
{
  float value[6];
};

/**
 * @brief CUDA kernel for preprocessing an image.
 *
 * This kernel takes an input image in uint8_t format and performs preprocessing operations
 * to convert it into a float format. The image is resized to the specified destination width
 * and height, and the color channels are rearranged. The resulting image is stored in the
 * destination array.
 *
 * @param src Pointer to the input image data in uint8_t format.
 * @param dst Pointer to the destination array for the preprocessed image data in float format.
 * @param dst_width The width of the destination image.
 * @param dst_height The height of the destination image.
 * @param edge The number of elements to process in the input image.
 */
__global__ void preprocess_kernel(
    uint8_t *src, float *dst, int dst_width,
    int dst_height, int edge)
{
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= edge)
      return;

  int dx = position % dst_width; 
  int dy = position / dst_width; 

  // float c0 = src[dy * dst_width * 3 + dx * 3 + 0] / 255.0f;
  // float c1 = src[dy * dst_width * 3 + dx * 3 + 1] / 255.0f;
  // float c2 = src[dy * dst_width * 3 + dx * 3 + 2] / 255.0f;

  int src_index = dy * dst_width * 3 + dx * 3;
  float c0 = src[src_index] / 255.0f;
  float c1 = src[src_index + 1] / 255.0f;
  float c2 = src[src_index + 2] / 255.0f;

  float t = c2;
  c2 = c0;
  c0 = t;

  int area = dst_width * dst_height;
  float *pdst_c0 = dst + dy * dst_width + dx;
  float *pdst_c1 = pdst_c0 + area;
  float *pdst_c2 = pdst_c1 + area;
  *pdst_c0 = c0;
  *pdst_c1 = c1;
  *pdst_c2 = c2;
}


/**
 * @brief Applies an affine transformation to an input image and stores the result in the output image.
 *
 * This kernel function performs an affine transformation on the input image using the provided affine matrix.
 * The transformed image is stored in the output image.
 *
 * @param src               Pointer to the input image data.
 * @param src_line_size     The line size of the input image.
 * @param src_width         The width of the input image.
 * @param src_height        The height of the input image.
 * @param dst               Pointer to the output image data.
 * @param dst_width         The width of the output image.
 * @param dst_height        The height of the output image.
 * @param const_value_st    The constant value used for out-of-bounds pixels.
 * @param d2s               The affine matrix used for the transformation.
 * @param edge              The number of pixels to process.
 */
__global__ void warpaffine_kernel(
    uint8_t *src, int src_line_size, int src_width,
    int src_height, float *dst, int dst_width,
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int edge)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge)
        return;

    // Calculate destination pixel indices
    int dx = position % dst_width;
    int dy = position / dst_width;

    // Compute source coordinates
    float src_x = d2s.value[0] * dx + d2s.value[1] * dy + d2s.value[2] + 0.5f;
    float src_y = d2s.value[3] * dx + d2s.value[4] * dy + d2s.value[5] + 0.5f;

    // Initialize color components with the constant value for out-of-bound cases
    float c0, c1, c2;
    c0 = c1 = c2 = const_value_st;

    // Check if the computed source coordinates are within the bounds
    if (src_x > -1 && src_x < src_width && src_y > -1 && src_y < src_height) {
        int x_low = max(0, min(static_cast<int>(floorf(src_x)), src_width - 1));
        int y_low = max(0, min(static_cast<int>(floorf(src_y)), src_height - 1));
        int x_high = min(x_low + 1, src_width - 1);
        int y_high = min(y_low + 1, src_height - 1);

        // Compute bilinear interpolation weights
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        // Fetch pixel values for interpolation
        uint8_t *v1 = src + y_low * src_line_size + x_low * 3;
        uint8_t *v2 = src + y_low * src_line_size + x_high * 3;
        uint8_t *v3 = src + y_high * src_line_size + x_low * 3;
        uint8_t *v4 = src + y_high * src_line_size + x_high * 3;

        // Perform bilinear interpolation
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    // Convert BGR to RGB and normalize
    float t = c0; // Swap c0 and c2 to convert BGR to RGB
    c0 = c2 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = t / 255.0f;

    // Write to destination: rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    dst[dy * dst_width + dx] = c0;
    dst[area + dy * dst_width + dx] = c1;
    dst[2 * area + dy * dst_width + dx] = c2;
}


/**
 * @brief Performs CUDA-based pure preprocessing on an input image.
 *
 * This function copies the input image from the host to the device, and then launches a CUDA kernel
 * to perform the preprocessing operation on the image. The resulting preprocessed image is stored
 * in the destination buffer.
 *
 * @param src The input image buffer in uint8_t format.
 * @param dst The destination buffer to store the preprocessed image in float format.
 * @param dst_width The width of the destination image.
 * @param dst_height The height of the destination image.
 */
void cuda_pure_preprocess(
    uint8_t *src, float *dst, int dst_width, int dst_height)
{

  int img_size = dst_width * dst_height * 3;
  CUDA_CHECK(cudaMemcpy(img_buffer_device, src, img_size, cudaMemcpyHostToDevice));

  int jobs = dst_height * dst_width;
  int threads = 256;
  int blocks = (jobs + threads - 1) / threads;

  preprocess_kernel<<<blocks, threads>>>(
      img_buffer_device, dst, dst_width, dst_height, jobs);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }

}

/**
 * @brief Applies a preprocessing step on the input image using CUDA.
 *
 * This function takes an input image in the form of a uint8_t array and applies a series of transformations
 * to resize and warp the image. The resulting image is stored in a float array.
 *
 * @param src The input image array in uint8_t format.
 * @param src_width The width of the input image.
 * @param src_height The height of the input image.
 * @param dst The output image array in float format.
 * @param dst_width The desired width of the output image.
 * @param dst_height The desired height of the output image.
 */
void cuda_preprocess(
    uint8_t *src, int src_width, int src_height,
    float *dst, int dst_width, int dst_height)
{

  int img_size = src_width * src_height * 3;
  CUDA_CHECK(cudaMemcpy(img_buffer_device, src, img_size, cudaMemcpyHostToDevice));

  AffineMatrix s2d, d2s;
  float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

  s2d.value[0] = scale;
  s2d.value[1] = 0;
  s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
  s2d.value[3] = 0;
  s2d.value[4] = scale;
  s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

  cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
  cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
  cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

  memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

  int jobs = dst_height * dst_width;
  int threads = 256;
  int blocks = (jobs + threads - 1) / threads;

  warpaffine_kernel<<<blocks, threads>>>(
      img_buffer_device, src_width * 3, src_width,
      src_height, dst, dst_width,
      dst_height, 128, d2s, jobs);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
}

/**
 * @brief Preprocesses a batch of images on the GPU.
 *
 * This function takes a batch of images represented as OpenCV Mats and preprocesses them on the GPU.
 * The preprocessed images are stored in a float array.
 *
 * @param img_batch The batch of input images.
 * @param dst The destination array to store the preprocessed images.
 * @param dst_width The width of the preprocessed images.
 * @param dst_height The height of the preprocessed images.
 */
void cuda_batch_preprocess(std::vector<cv::Mat> &img_batch,
                           float *dst, int dst_width, int dst_height)
{
  int dst_size = dst_width * dst_height * 3;
  for (size_t i = 0; i < img_batch.size(); i++) {
    cuda_preprocess(img_batch[i].ptr(), img_batch[i].cols, img_batch[i].rows, &dst[dst_size * i], dst_width, dst_height);
  }
}

/**
 * @brief Initializes the GPU preprocessing.
 *
 * This function allocates memory on the GPU for the input image buffer.
 *
 * @param max_image_size The maximum size of the input image.
 */
void cuda_preprocess_init(int max_image_size)
{
    // prepare input data in device memory
    size_t total_bytes = max_image_size * 3 * sizeof(uint8_t); 
    CUDA_CHECK(cudaMalloc((void **)&img_buffer_device, total_bytes));
}

/**
 * @brief Cleans up the GPU preprocessing.
 *
 * This function frees the memory allocated on the GPU for the input image buffer.
 */
void cuda_preprocess_destroy()
{
  CUDA_CHECK(cudaFree(img_buffer_device));
}



/**
 * @brief Process the input image on the GPU.
 *
 * This function takes an input image, performs preprocessing on it, and stores the result in the input_device_buffer.
 *
 * @param src The input image to be processed.
 * @param input_device_buffer The device buffer to store the preprocessed image.
 * @param method The scaling method to be used for preprocessing.
 *
 * @throws std::invalid_argument if an unsupported scale method is provided.
 */
void process_input_gpu(cv::Mat &src, float *input_device_buffer, ScaleMethod method)
{
  switch (method) {
    case ScaleMethod::Resize:
      {
        cv::Mat warp_dst = resize_image(src, kInputW, kInputH);
        cuda_pure_preprocess(warp_dst.ptr<unsigned char>(), input_device_buffer, kInputW, kInputH);
      }
      break;
    case ScaleMethod::LetterBox:
      cuda_preprocess(src.ptr<unsigned char>(), src.cols, src.rows, input_device_buffer, kInputW, kInputH);
      break;
    default:
      throw std::invalid_argument("Unsupported scale method");
  }
}


/**
 * Applies letterbox transformation to the input image.
 * 
 * @param src The input image to be letterboxed.
 * @return The letterboxed image.
 */
inline cv::Mat letterbox(cv::Mat &src)
{
  float scale = std::min(kInputH / (float)src.rows, kInputW / (float)src.cols);

  int offsetx = static_cast<int>((kInputW - src.cols * scale) / 2);
  int offsety = static_cast<int>((kInputH - src.rows * scale) / 2);

  cv::Point2f srcTri[3] = {
      cv::Point2f(0.f, 0.f),
      cv::Point2f(src.cols - 1.f, 0.f),
      cv::Point2f(0.f, src.rows - 1.f)
  };
  cv::Point2f dstTri[3] = {
      cv::Point2f(offsetx, offsety),
      cv::Point2f(src.cols * scale - 1.f + offsetx, offsety),
      cv::Point2f(offsetx, src.rows * scale - 1.f + offsety)
  };

  cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
  cv::Mat warp_dst = cv::Mat::zeros(kInputH, kInputW, src.type());
  cv::warpAffine(src, warp_dst, warp_mat, warp_dst.size());
  
  return warp_dst;
}



/**
 * @brief Applies a preprocessing step to the input image based on the specified scale method.
 *
 * This function takes an input image and applies a preprocessing step based on the specified scale method.
 * The supported scale methods are Resize and LetterBox.
 *
 * @param src The input image to be processed.
 * @param input_device_buffer The device buffer to store the preprocessed image.
 * @param method The scale method to be used for preprocessing.
 *
 * @throws std::invalid_argument if an unsupported scale method is provided.
 */
void process_input_cv_affine(cv::Mat &src, float *input_device_buffer, ScaleMethod method)
{
  switch (method) {
    case ScaleMethod::Resize:
      {
        cv::Mat dst;
        cv::resize(src, dst, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_LINEAR);
        cuda_pure_preprocess(dst.ptr<unsigned char>(), input_device_buffer, kInputW, kInputH);
      }
      break;
    case ScaleMethod::LetterBox:
      {
        auto warp_dst = letterbox(src);
        cuda_pure_preprocess(warp_dst.ptr(), input_device_buffer, kInputW, kInputH);
      }
      break;
    default:
      throw std::invalid_argument("Unsupported scale method");
  }
}



/**
 * @brief Processes the input image on the CPU.
 *
 * This function performs preprocessing on the input image before feeding it to the GPU for further processing.
 * The input image is resized or letterboxed based on the specified scale method. The image is then normalized
 * and converted from BGR to RGB color space. Finally, the image is converted from NHWC to NCHW format and
 * copied to the device memory.
 *
 * @param src The input image to be processed.
 * @param input_device_buffer The device buffer to store the processed image.
 * @param method The scale method to be used for preprocessing.
 *
 * @throws std::invalid_argument if an unsupported scale method is provided.
 */
void process_input_cpu(cv::Mat &src, float *input_device_buffer, ScaleMethod method)
{
  // Perform resizing or letterboxing based on the scale method
  cv::Mat warp_dst;
  switch (method) {
    case ScaleMethod::Resize:
      {
        cv::resize(src, warp_dst, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_LINEAR);
      }
      break;
    case ScaleMethod::LetterBox:
      {
        warp_dst = letterbox(src);                    // letterbox
      }
      break;
    default:
      throw std::invalid_argument("Unsupported scale method");
  }
  
  warp_dst.convertTo(warp_dst, CV_32FC3, 1.0 / 255.0); // normalization
  cv::cvtColor(warp_dst, warp_dst, cv::COLOR_BGR2RGB); // BGR2RGB

  // NHWC to NCHW：rgbrgbrgb to rrrgggbbb：
  std::vector<cv::Mat> warp_dst_nchw_channels;
  cv::split(warp_dst, warp_dst_nchw_channels);

  for (auto &img : warp_dst_nchw_channels) {
    img = img.reshape(1, 1);
  }
  
  cv::Mat warp_dst_nchw;
  cv::hconcat(warp_dst_nchw_channels, warp_dst_nchw);

  CUDA_CHECK(cudaMemcpy(input_device_buffer, warp_dst_nchw.ptr(), kInputH * kInputW * 3 * sizeof(float), cudaMemcpyHostToDevice));
}


__device__ float lerp(float a, float b, float t) {
  return a + t * (b - a);
}

__global__ void resize_kernel(unsigned char* input, unsigned char* output, int original_width, int original_height, int new_width, int new_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= new_width || y >= new_height) return;

  float gx = ((float)x + 0.5) * ((float)original_width / (float)new_width) - 0.5;
  float gy = ((float)y + 0.5) * ((float)original_height / (float)new_height) - 0.5;

  int gxi = (int)gx;
  int gyi = (int)gy;

  unsigned char result[3] = {0, 0, 0};

  for (int channel = 0; channel < 3; ++channel) {
      float c00 = input[(gyi * original_width + gxi) * 3 + channel];
      float c10 = input[(gyi * original_width + min(gxi + 1, original_width - 1)) * 3 + channel];
      float c01 = input[(min(gyi + 1, original_height - 1) * original_width + gxi) * 3 + channel];
      float c11 = input[(min(gyi + 1, original_height - 1) * original_width + min(gxi + 1, original_width - 1)) * 3 + channel];

      float cx0 = lerp(c00, c10, gx - gxi);
      float cx1 = lerp(c01, c11, gx - gxi);

      result[channel] = (unsigned char)lerp(cx0, cx1, gy - gyi);
  }

  int index = (y * new_width + x) * 3;
  output[index] = result[0];
  output[index + 1] = result[1];
  output[index + 2] = result[2];
}

cv::Mat resize_image(cv::Mat &src, int new_width, int new_height) {
  const size_t input_size = src.cols * src.rows * src.channels();
  const size_t output_size = new_width * new_height * src.channels();
  unsigned char *d_input, *d_output;

  cudaMalloc(&d_input, input_size);
  cudaMalloc(&d_output, output_size);

  cudaMemcpy(d_input, src.ptr(), input_size, cudaMemcpyHostToDevice);

  dim3 block_size(16, 16);
  dim3 num_blocks((new_width + block_size.x - 1) / block_size.x, (new_height + block_size.y - 1) / block_size.y);

  resize_kernel<<<num_blocks, block_size>>>(d_input, d_output, src.cols, src.rows, new_width, new_height);
  cudaDeviceSynchronize();

  cv::Mat dst(new_height, new_width, src.type());
  cudaMemcpy(dst.ptr(), d_output, output_size, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);

  return dst;
}
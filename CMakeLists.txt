cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC") # Ensure -fPIC is used for CUDA

project(xtrt LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -g -Wall -Werror -Wno-error=unused-variable -Wno-error=unused-function -fPIC")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

option(BUILD_TOOLS_EVAL "Build eval tool" ON)

if(BUILD_TOOLS_EVAL)
    add_subdirectory(tools/eval)
endif()

add_library(yolo_plugin SHARED
    plugins/YoloLayer_TRT/yoloPlugins.cpp
    plugins/YoloLayer_TRT/yoloForward_nc.cu
)

add_library(yolo_utils SHARED
    common/postprocess.cpp
    common/preprocess.cu
)

add_executable(build
    tools/deploy/build/build.cu
    tools/deploy/build/calibrator.cpp
    common/logger.cpp
)

add_executable(yolo_det_img
    tools/deploy/infer/yolo_det_img.cu
    common/logger.cpp
)

add_executable(yolo_det
    tools/deploy/infer/yolo_det.cu
    common/logger.cpp
)

target_link_libraries(yolo_plugin  PUBLIC ${TensorRT_LIBRARIES})
target_link_libraries(yolo_utils   PUBLIC ${OpenCV_LIBS})
target_link_libraries(build        PRIVATE -Wl,--no-as-needed yolo_utils yolo_plugin) # -Wl,--no-as-needed is needed to avoid linking errors
target_link_libraries(yolo_det     PRIVATE yolo_utils yolo_plugin)
target_link_libraries(yolo_det_img PRIVATE yolo_utils  yolo_plugin)

target_include_directories(yolo_plugin  PUBLIC ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_include_directories(yolo_utils   PRIVATE ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} common config)
target_include_directories(build        PRIVATE ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} common config)
target_include_directories(yolo_det_img PRIVATE ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} common config)
target_include_directories(yolo_det     PRIVATE ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} common config)


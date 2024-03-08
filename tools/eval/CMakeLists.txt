cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC") # Ensure -fPIC is used for CUDA

project(eval LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -g -Wall -Werror -Wno-error=unused-variable -Wno-error=unused-function -fPIC")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

option(BUILD_TOOLS_EVAL_INDEPENDENT "Build eval tool independently" OFF)

if(BUILD_TOOLS_EVAL_INDEPENDENT)
    add_library(yolo_plugin SHARED
        ../../plugins/YoloLayer_TRT/yoloPlugins.cpp
        ../../plugins/YoloLayer_TRT/yoloForward_nc.cu
    )

    add_library(yolo_utils SHARED
        ../../common/postprocess.cpp
        ../../common/preprocess.cu
    )
endif()

add_executable(yolo_eval
    main.cpp
    yolo_infer.cpp
    ../../common/logger.cpp
)

add_executable(profile
    profile.cpp
    ../../common/logger.cpp
)

target_include_directories(yolo_eval PRIVATE ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ../../common ../../config)
target_include_directories(profile    PRIVATE ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ../../common ../../config)

target_link_libraries(yolo_eval PRIVATE yolo_plugin yolo_utils ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
target_link_libraries(profile    PRIVATE yolo_plugin yolo_utils ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})


cmake_minimum_required(VERSION 3.11)
project(xtrt LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)

add_library(yolo_plugin SHARED
    plugins/YoloLayer_TRT/yoloPlugins.cpp
    plugins/YoloLayer_TRT/yoloForward_nc.cu
)
target_include_directories(yolo_plugin PUBLIC ${TensorRT_INCLUDE_DIRS})
target_link_libraries(yolo_plugin PUBLIC ${TensorRT_LIBRARIES})
set_target_properties(yolo_plugin PROPERTIES CUDA_ARCHITECTURES "61;70;75")

add_library(yolo_utils SHARED
    common/postprocess.cpp
    common/preprocess.cu
)
target_include_directories(yolo_utils PRIVATE ${TensorRT_INCLUDE_DIRS} config)
target_link_libraries(yolo_utils PUBLIC ${OpenCV_LIBS})
set_target_properties(yolo_utils PROPERTIES CUDA_ARCHITECTURES "61;70;75")

add_executable(build
    xbuild/build.cu
    xbuild/calibrator.cpp
    common/logger.cpp
)
target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS} common)
target_link_libraries(build PRIVATE -Wl,--no-as-needed yolo_plugin) # -Wl,--no-as-needed is needed to avoid linking errors
target_link_libraries(build PRIVATE yolo_utils)
set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75")

add_executable(yolo_det_img
    xinfer/yolo_det_img.cu
    common/logger.cpp
)
target_include_directories(yolo_det_img PRIVATE ${TensorRT_INCLUDE_DIRS} common config)
target_link_libraries(yolo_det_img PRIVATE -Wl,--no-as-needed yolo_plugin)
target_link_libraries(yolo_det_img PRIVATE yolo_utils)
set_target_properties(yolo_det_img PROPERTIES CUDA_ARCHITECTURES "61;70;75")

add_executable(yolo_det
    xinfer/yolo_det.cu
    common/logger.cpp
)
target_include_directories(yolo_det PRIVATE ${TensorRT_INCLUDE_DIRS} common config)
target_link_libraries(yolo_det PRIVATE -Wl,--no-as-needed yolo_plugin)
target_link_libraries(yolo_det PRIVATE yolo_utils)
set_target_properties(yolo_det PROPERTIES CUDA_ARCHITECTURES "61;70;75")

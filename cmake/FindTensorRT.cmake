## find tensorrt
include(FindPackageHandleStandardArgs)

# Allow users to specify the TensorRT search path
set(TensorRT_ROOT
	""
	CACHE
	PATH
	"TensorRT root directory")

# Define the paths to search for TensorRT
set(TensorRT_SEARCH_PATH
  /usr/src/tensorrt
  /usr/include/x86_64-linux-gnu
  /usr/include/aarch64-linux-gnu
  /usr/lib/x86_64-linux-gnu
  /usr/lib/aarch64-linux-gnu
  ${TensorRT_ROOT}
)

# Specify the TensorRT libraries to search for
set(TensorRT_ALL_LIBS
  nvinfer
  nvinfer_plugin
  nvparsers
  nvonnxparser
)

# Predefine variables for later use
set(TensorRT_LIBS_LIST)
set(TensorRT_LIBRARIES)

# Search for the TensorRT include directory
find_path(
  TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS ${TensorRT_SEARCH_PATH}
)

# Use the version file located in the include directory to set the TensorRT version
if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

  string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
  set(TensorRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()
message("TensorRT version: ${TensorRT_VERSION_STRING}")

# Search for the path containing sample code
find_path(
  TensorRT_SAMPLE_DIR
  NAMES trtexec/trtexec.cpp
  PATHS ${TensorRT_SEARCH_PATH}
  PATH_SUFFIXES samples
)

# Search for each specified TensorRT library
foreach(lib ${TensorRT_ALL_LIBS} )
  find_library(
    TensorRT_${lib}_LIBRARY
    NAMES ${lib}
    PATHS ${TensorRT_SEARCH_PATH}
  )
  # Store found library variables for later use
  set(TensorRT_LIBS_VARS TensorRT_${lib}_LIBRARY ${TensorRT_LIBS_LIST})
  ## 也是TensorRT的依赖库，存成list，方便后面用foreach
  list(APPEND TensorRT_LIBS_LIST TensorRT_${lib}_LIBRARY)
endforeach()

# Utilize CMake's built-in functionality to set standard variables like xxx_FOUND
find_package_handle_standard_args(TensorRT REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_SAMPLE_DIR ${TensorRT_LIBS_VARS})

if(TensorRT_FOUND)
  # Populate the TensorRT_LIBRARIES variable
  foreach(lib ${TensorRT_LIBS_LIST} )
    list(APPEND TensorRT_LIBRARIES ${${lib}})
  endforeach()
  message("Found TensorRT: ${TensorRT_INCLUDE_DIR} ${TensorRT_LIBRARIES} ${TensorRT_SAMPLE_DIR}")
  message("TensorRT version: ${TensorRT_VERSION_STRING}")
endif()
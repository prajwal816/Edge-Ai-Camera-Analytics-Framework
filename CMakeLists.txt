cmake_minimum_required(VERSION 3.18)
project(edge_infer_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(EDGE_USE_TENSORRT "Enable TensorRT headers (requires TensorRT SDK in include/lib paths)" OFF)
option(EDGE_USE_CUDA "Enable CUDA preprocessing path (requires CUDA toolkit)" OFF)

if(EDGE_USE_TENSORRT)
  add_compile_definitions(EDGE_USE_TENSORRT=1)
endif()
if(EDGE_USE_CUDA)
  add_compile_definitions(EDGE_USE_CUDA=1)
endif()

set(EDGE_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp")

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

add_library(
  edge_infer_core STATIC
  ${EDGE_CPP_ROOT}/inference/inference_engine.cpp
  ${EDGE_CPP_ROOT}/inference/edge_service.cpp
  ${EDGE_CPP_ROOT}/preprocessing/preprocessor.cpp
  ${EDGE_CPP_ROOT}/scheduler/dynamic_scheduler.cpp
  ${EDGE_CPP_ROOT}/cuda_utils/gpu_metrics.cpp
)

target_include_directories(edge_infer_core PUBLIC ${EDGE_CPP_ROOT})

pybind11_add_module(edge_infer_native ${EDGE_CPP_ROOT}/bindings/python_bindings.cpp)
target_link_libraries(edge_infer_native PRIVATE edge_infer_core)

set_target_properties(edge_infer_native PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                  "${CMAKE_BINARY_DIR}/python_module")

install(TARGETS edge_infer_native LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

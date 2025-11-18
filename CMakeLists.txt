cmake_minimum_required (VERSION 3.23)
project ("CudaKernels" C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(BUILD_PYTHON "Build torch api and pybindings" OFF)

if(BUILD_PYTHON)
 find_package(Python REQUIRED COMPONENTS Interpreter Development)
 find_package(Torch REQUIRED)
 find_package(pybind11 CONFIG REQUIRED)
endif()

add_subdirectory(kernels)
add_subdirectory(cpp_samples)
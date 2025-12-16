file(GLOB TORCH_API_SOURCE "src/*.cu")

add_library(cuda_torch STATIC ${TORCH_API_SOURCE})
target_include_directories(cuda_torch PUBLIC include)
target_link_libraries(cuda_torch PRIVATE cuda_kernels ${TORCH_LIBRARIES})

add_subdirectory(python)
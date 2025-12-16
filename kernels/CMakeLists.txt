file(GLOB KERNELS_SOURCE "src/*.cu")

add_library(cuda_kernels STATIC ${KERNELS_SOURCE})
target_include_directories(cuda_kernels PUBLIC include)
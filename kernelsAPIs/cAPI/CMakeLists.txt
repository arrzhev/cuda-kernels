file(GLOB C_API_SOURCE "src/*.cu")

add_library(cuda_c SHARED ${C_API_SOURCE})
target_include_directories(cuda_c PUBLIC include)
target_link_libraries(cuda_c PRIVATE cuda_kernels)
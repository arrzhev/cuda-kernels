
file(GLOB KERNELS_SOURCE "src/*.cu")
file(GLOB C_API_SOURCE "cAPI/*.cu")

add_library(cuda_ops SHARED ${KERNELS_SOURCE} ${C_API_SOURCE})
target_include_directories(cuda_ops
 PRIVATE
 src/include
 PUBLIC
 cAPI/include
)

if(BUILD_PYTHON)
 file(GLOB TORCH_API_SOURCE "torchAPI/*.cu")

 add_library(torch_extension_c STATIC ${KERNELS_SOURCE} ${TORCH_API_SOURCE})
 target_include_directories(torch_extension_c
  PRIVATE
  src/include
  PUBLIC
  torchAPI/include
 )
 target_link_libraries(torch_extension_c PUBLIC ${TORCH_LIBRARIES})

 add_subdirectory(python)
endif()
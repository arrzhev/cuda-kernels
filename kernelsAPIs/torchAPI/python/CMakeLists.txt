# Create python module with name 'torch_extension'
pybind11_add_module(torch_extension binder.cpp)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(torch_extension
 PRIVATE
 cuda_torch
 ${TORCH_PYTHON_LIBRARY}
 ${TORCH_LIBRARIES}
)

install(TARGETS torch_extension
 COMPONENT python
 LIBRARY DESTINATION "${PYTHON_MODULE_INSTALL_DIR}")
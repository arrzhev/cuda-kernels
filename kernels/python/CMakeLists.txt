# Create python module with name 'torch_extension'
pybind11_add_module(torch_extension binder.cpp)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(torch_extension
 PRIVATE
 torch_extension_c
 ${TORCH_PYTHON_LIBRARY}
)

install(TARGETS torch_extension
 COMPONENT python
 LIBRARY DESTINATION "${PYTHON_MODULE_INSTALL_DIR}")
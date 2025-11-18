# CUDA Kernels

Set of popular CUDA kernel functions with various optimizations accompanied by simple C and Torch APIs to demonstrate performance.

## Getting Started

### Prerequisites

1. Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
2. Download and install CMake.
3. To be able to run python samples following packages are needed:
    - pyTorch with cuda support (same version as your installed cuda toolkit)
    - pybind11
    - pytest
    - scikit-image
    - matplotlib

### Installation process

1.  Clone the repository:
    ```
    git clone https://github.com/arrzhev/cuda-kernels.git
    ```

2. Navigate to the root of the cloned repository and create a build directory:
    ```
    mkdir build && cd build
    ```

3. Configure the project with CMake:
    ```
    cmake ..
    ```

    If you want to build python samples, use the following flags:
    - BUILD_PYTHON
    - PYTHON_MODULE_INSTALL_DIR

    Example:
    ```
    cmake -DBUILD_PYTHON=ON -DPYTHON_MODULE_INSTALL_DIR="path/to/your/python/lib/site-packages/" ..
    ```
    
4. Build the project:
    ```
    cmake --build .
    ```

### Use cases

1. Test each kernel via C API:

    Example of how to run VectorAdd kernel from C API on Linux
    ```
    build/cpp_samples/vectorAdd
    ```

2. Test kernels via pytest and PyTorch API:
    ```
    pytest -s
    ```
# CUDA Kernels

Set of CUDA kernel functions with various optimizations accompanied by simple C and PyTorch APIs as well as unit and performance tests with api usage examples.

## Project structure

- `kernels/`: Main cuda kernels directory
    - `include/`: Cuda kernels public headers
    - `src/`: Cuda kernels source code
- `kernelsAPIs/`: Available kernels APIs for convenient usage
    - `cAPI/`: C/C++ APIs for each kernel 
        - `include/`: C/C++ APIs public headers
        - `src/`: C/C++ APIs source code
    - `torchAPI/`: PyTorch C/C++ APIs for each kernel
        - `include/`: PyTorch C/C++ APIs public headers
        - `src/`: PyTorch C/C++ APIs source code
        - `python/`: PyBindings for PyTorch C/C++ kernels APIs
- `tests/`: Unit and performance tests for each kernel. Uses python extension
- `cpp_samples/`: Sample use cases for each kernel using C/C++ API
- `python_samples/`: Sample use cases for each kernel using python extension

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

## Performance overview of Matmul kernels

Examples of achieved performance running kernels on NVIDIA Tesla T4 (Turing).

1. Matmul shape 2048x2048x2048, all matrices have row-major layout and fp32 data type:

<!-- benchmark_results -->
| Kernel                                       |  TFLOPs/s | Performance compared to PyTorch|
|:---------------------------------------------|:---------:|:------------------------------:|
| 0: PyTorch                                   |   `3.57`  | 100%                           |
| 1: Naive                                     |   `0.06`  | 1.57%                          |
| 2: Coalescing                                |   `0.47`  | 16.13%                         |
| 3: Block tiles                               |   `0.79`  | 22.69%                         |
| 4: 1D Thread tiles                           |   `2.05`  | 62.30%                         |
| 5: 2D Thread tiles                           |   `2.90`  | 87.97%                         |
| 7: 2D Thread tiles vectorized                |   `3.39`  | 94.72%                         |
| 8: 2D Thread tiles vectorized double buffer  |   `3.36`  | 94.40%                         |
<!-- benchmark_results -->

2. Single optimized kernel performance on various matrices shapes, all matrices have row-major layout and fp32 data type:

<!-- benchmark_results -->
| Shape (MxNxK)    | My kernel TFLOPs/s | PyTorch TFLOPs/s | Performance compared to PyTorch|
|:-----------------|:------------------:|:----------------:|:------------------------------:|
| 1: 128x128x128   |       `0.13`       |      `0.20`      | 62.99%                         |
| 2: 256x256x256   |       `0.96`       |      `1.11`      | 87.68%                         |
| 3: 512x512x512   |       `2.19`       |      `3.81`      | 57.57%                         |
| 4: 1024x1024x1024|       `3.29`       |      `3.80`      | 86.65%                         |
| 5: 2048x2048x2048|       `3.38`       |      `3.65`      | 92.51%                         |
| 6: 512x4068x7168 |       `3.18`       |      `4.30`      | 73.97%                         |
| 7: 512x7168x2304 |       `3.44`       |      `4.33`      | 79.53%                         |
| 8: 512x512x1024  |       `2.10`       |      `3.80`      | 55.20%                         |
| 9: 512x512x2048  |       `2.66`       |      `3.95`      | 67.34%                         |
|10: 512x512x4096  |       `2.90`       |      `4.05`      | 71.65%                         |
|11: 512x512x8192  |       `2.98`       |      `3.90`      | 76.41%                         |
<!-- benchmark_results -->
# CUDA Kernels

Set of CUDA kernel functions with various optimizations accompanied by simple C and PyTorch APIs as well as unit and performance tests with api usage examples.

## Project structure

- `kernels/`: Main cuda kernels directory with kernel's source code
- `kernelsAPIs/`: Available kernels APIs for convenient usage
    - `cAPI/`: C/C++ APIs for each kernel 
    - `torchAPI/`: PyTorch C/C++ APIs for each kernel
        - `python/`: PyBindings for PyTorch C/C++ kernels APIs
- `tests/`: Unit and performance tests for each kernel. Uses python extension
- `cpp_samples/`: Sample use cases for each kernel using C/C++ API
- `python_samples/`: Sample use cases for each kernel using python extension

## Getting Started

### Prerequisites

1. Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your platform.
2. Download and install CMake.
3. To be able to compile pyTorchApi and run test and python samples following packages are needed:
    - pyTorch with cuda support (same version as your installed cuda toolkit)
    - pybind11
    - pytest
    - scikit-image
    - matplotlib

**Note:** If you are using **cuda 12.6** feel free to install requirements, otherwise install pyTorch with needed cuda version support.

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
| Id |   M   |   N   |   K   | My kernel TFLOPs/s | PyTorch TFLOPs/s | Performance compared to PyTorch|
|:---|:-----:|:-----:|:-----:|:------------------:|:----------------:|:------------------------------:|
| 1  | 128   | 128   | 128   |       `0.13`       |      `0.20`      | 62.99%                         |
| 2  | 256   | 256   | 256   |       `0.96`       |      `1.11`      | 87.68%                         |
| 3  | 512   | 512   | 512   |       `2.19`       |      `3.81`      | 57.57%                         |
| 4  | 1024  | 1024  | 1024  |       `3.29`       |      `3.80`      | 86.65%                         |
| 5  | 2048  | 2048  | 2048  |       `3.38`       |      `3.65`      | 92.51%                         |
| 6  | 512   | 4068  | 7168  |       `3.18`       |      `4.30`      | 73.97%                         |
| 7  | 512   | 7168  | 2304  |       `3.44`       |      `4.33`      | 79.53%                         |
| 8  | 512   | 512   | 1024  |       `2.10`       |      `3.80`      | 55.20%                         |
| 9  | 512   | 512   | 2048  |       `2.66`       |      `3.95`      | 67.34%                         |
| 10 | 512   | 512   | 4096  |       `2.90`       |      `4.05`      | 71.65%                         |
| 11 | 512   | 512   | 8192  |       `2.98`       |      `3.90`      | 76.41%                         |
<!-- benchmark_results -->
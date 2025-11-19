#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <tensorMulTorch.hpp>
#include <matrixMul.cuh>
#include <util.cuh>

#include "common.cuh"

torch::Tensor tensorMul(torch::Tensor x, torch::Tensor y, auto matrixVectorMulKernel)
{
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    const auto xDim = x.dim();
    const auto yDim = y.dim();
    TORCH_CHECK(xDim > 0 && yDim > 0, "x and y tensors must have the same size");

    torch::Tensor z;

    if(xDim == 1 && yDim == 1)
    {
        TORCH_CHECK(x.size(0) == y.size(0), "x and y tensors are 1D and must have the same size");
        TORCH_CHECK(false, "Vector dot product is not implemented yet");
    }
    else if(xDim == 2 && yDim == 1)
    {
        TORCH_CHECK(x.size(1) == y.size(0), "x and y tensors must have the same cols count");
        
        int rows = x.size(0);
        int cols = x.size(1);
        z = torch::empty(rows, x.options());

        matrixVectorMulKernel(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), rows, cols);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    else
        TORCH_CHECK(false, "Not supported dimensions");

    return z;
}

torch::Tensor tensorMulNaive(torch::Tensor x, torch::Tensor y)
{
    auto matrixVectorMulKernel = [](const float *X, const float *y, float *z, unsigned rows, unsigned cols)
    {
        const int blockSize = 256;
        const int gridSize = (rows + blockSize - 1) / blockSize;

        matrixVectorMul_naive_kernel<<<gridSize, blockSize>>>(X, y, z, rows, cols);
    };
    
    return tensorMul(x, y, matrixVectorMulKernel);
}

torch::Tensor tensorMulShared(torch::Tensor x, torch::Tensor y)
{
    auto matrixVectorMulKernel = [](const float *X, const float *y, float *z, unsigned rows, unsigned cols)
    {
        const int blockSize = 256;
        const int gridSize = rows;
        const size_t sharedSize = blockSize * sizeof(double);

        matrixVectorMul_shared_kernel<<<gridSize, blockSize, sharedSize>>>(X, y, z, rows, cols);
    };

    return tensorMul(x, y, matrixVectorMulKernel);
}

torch::Tensor tensorMulWarp(torch::Tensor x, torch::Tensor y)
{
    auto matrixVectorMulKernel = [](const float *X, const float *y, float *z, unsigned rows, unsigned cols)
    {
        const int blockSize = 256;
        const int gridSize = rows;
        const size_t sharedSize = CEIL_DIV(blockSize, 32) * sizeof(double);

        matrixVectorMul_warp_kernel<<<gridSize, blockSize, sharedSize>>>(X, y, z, rows, cols);
    };

    return tensorMul(x, y, matrixVectorMulKernel);
}
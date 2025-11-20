#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <matrixVectorMulTorch.hpp>
#include <matrixVectorMul.cuh>
#include <util.cuh>

#include "common.cuh"

static void launchMatrixVectorNaiveKernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
{
    const unsigned blockDim = 256;
    const unsigned gridDim = CEIL_DIV(rows, blockDim);

    matrixVectorMul_naive_kernel<<<gridDim, blockDim>>>(X, y, z, rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
};

static void launchMatrixVectorSharedKernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
{
    const unsigned blockDim = 256;
    const unsigned gridDim = rows;
    const size_t sharedSize = blockDim * sizeof(double);

    matrixVectorMul_shared_kernel<<<gridDim, blockDim, sharedSize>>>(X, y, z, rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
};

static void launchMatrixVectorWarpKernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
{
    const unsigned blockDim = 256;
    const unsigned gridDim = rows;

    matrixVectorMul_warp_kernel<<<gridDim, blockDim>>>(X, y, z, rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
};

torch::Tensor matrixVectorMul_(torch::Tensor x, torch::Tensor y, auto launchMatrixVectorMulKernel)
{
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    TORCH_CHECK(x.dim() == 2 && y.dim() == 1, "x and y tensors must have dimensions equal to 1");
    TORCH_CHECK(x.size(1) == y.size(0), "x and y tensors must have the same cols count");
    
    const unsigned rows = x.size(0);
    const unsigned cols = x.size(1);
    auto z = torch::empty(rows, x.options());

    launchMatrixVectorMulKernel(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), rows, cols);

    return z;
}

torch::Tensor matrixVectorMul(torch::Tensor x, torch::Tensor y)
{
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    TORCH_CHECK(x.dim() == 2 && y.dim() == 1, "x and y tensors must have dimensions equal to 1");
    TORCH_CHECK(x.size(1) == y.size(0), "x and y tensors must have the same cols count");
    
    const unsigned rows = x.size(0);
    const unsigned cols = x.size(1);
    auto z = torch::empty(rows, x.options());

    const bool useOpt = rows < 128U || cols > 384U;
    if(useOpt)
        launchMatrixVectorWarpKernel(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), rows, cols);
    else
        launchMatrixVectorNaiveKernel(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), rows, cols);

    return z;
}

torch::Tensor matrixVectorMulNaive(torch::Tensor x, torch::Tensor y)
{   
    return matrixVectorMul_(x, y, launchMatrixVectorNaiveKernel);
}

torch::Tensor matrixVectorMulShared(torch::Tensor x, torch::Tensor y)
{
    return matrixVectorMul_(x, y, launchMatrixVectorSharedKernel);
}

torch::Tensor matrixVectorMulWarp(torch::Tensor x, torch::Tensor y)
{
    return matrixVectorMul_(x, y, launchMatrixVectorWarpKernel);
}
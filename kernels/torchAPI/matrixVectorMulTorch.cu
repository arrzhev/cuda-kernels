#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <matrixVectorMulTorch.hpp>
#include <matrixVectorMul.cuh>
#include <util.cuh>

#include "common.cuh"

torch::Tensor matrixVectorMul_(torch::Tensor x, torch::Tensor y, auto launchMatrixVectorMulKernel)
{
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_FP32(x);
    CHECK_FP32(y);
    TORCH_CHECK(x.dim() == 2 && y.dim() == 1, "x tensor must have 2 dimensions and y must have 1");
    TORCH_CHECK(x.size(1) == y.size(0), "x and y tensors must have the same cols count");
    
    const unsigned rows = x.size(0);
    const unsigned cols = x.size(1);
    auto z = torch::empty(rows, x.options());

    launchMatrixVectorMulKernel(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return z;
}

torch::Tensor matrixVectorMul(torch::Tensor x, torch::Tensor y)
{
    CHECK_FP32(x);
    CHECK_FP32(y);
    TORCH_CHECK(x.dim() == 2 && y.dim() == 1, "x tensor must have 2 dimensions and y must have 1");

    const unsigned rows = x.size(0);
    const unsigned cols = x.size(1);

    constexpr unsigned blockDim = 256U;
    const bool useOpt = rows < 128U || cols > 384U;
    auto launchKernel = useOpt ? launch_matrixVectorMul_warp<blockDim> : launch_matrixVectorMul_naive<blockDim>;

    return matrixVectorMul_(x, y, launchKernel);
}

torch::Tensor matrixVectorMulNaive(torch::Tensor x, torch::Tensor y)
{   
    constexpr unsigned blockDim = 256U;
    return matrixVectorMul_(x, y, launch_matrixVectorMul_naive<blockDim>);
}

torch::Tensor matrixVectorMulShared(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned blockDim = 256U;
    return matrixVectorMul_(x, y, launch_matrixVectorMul_shared<blockDim>);
}

torch::Tensor matrixVectorMulWarp(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned blockDim = 256U;
    return matrixVectorMul_(x, y, launch_matrixVectorMul_warp<blockDim>);
}
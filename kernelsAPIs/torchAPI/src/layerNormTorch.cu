#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <layerNormTorch.hpp>
#include <layerNorm.cuh>
#include <util.cuh>

#include "common.cuh"

torch::Tensor layerNorm(torch::Tensor X, torch::Tensor W, torch::Tensor B, float eps)
{
    CHECK_FP32(X);
    CHECK_FP32(W);
    CHECK_FP32(B);
    CHECK_INPUT(X);
    CHECK_INPUT(W);
    CHECK_INPUT(B);
    TORCH_CHECK(X.dim() == 2 && W.dim() == 1 && B.dim() == 1, "tensor X must be 2 dim, W and B 1");
    TORCH_CHECK(X.size(1) == W.size(0) && X.size(1) == B.size(0), "Tensors must have same size");

    const unsigned rows = X.size(0);
    const unsigned cols = X.size(1);

    auto Y = torch::empty({rows, cols}, X.options());

    constexpr unsigned blockDim = 256U;
    launch_layerNorm_naive<blockDim>(X.data_ptr<float>(), W.data_ptr<float>(), B.data_ptr<float>(), Y.data_ptr<float>(), eps, rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return Y;
}


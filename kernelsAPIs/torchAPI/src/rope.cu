#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <ropeTorch.hpp>
#include <rope.cuh>
#include <util.cuh>

#include "common.cuh"

torch::Tensor rope(torch::Tensor X)
{
    CHECK_FP32(X);
    CHECK_INPUT(X);
    TORCH_CHECK(X.dim() == 3, "tensor X must have 3 dims");

    const unsigned batchSize = X.size(0);
    const unsigned seqLen = X.size(1);
    const unsigned dim = X.size(2);

    TORCH_CHECK(dim % 2U == 0U, "Tensor X's last dimension must be even");

    auto Y = torch::empty({batchSize, seqLen, dim}, X.options());

    constexpr unsigned blockDim = 256U;
    launch_rope<blockDim>(X.data_ptr<float>(), Y.data_ptr<float>(), batchSize, seqLen, dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return Y;
}

torch::Tensor rope_cached(torch::Tensor X, torch::Tensor sinCosVec)
{
    CHECK_FP32(X);
    CHECK_FP32(sinCosVec);
    CHECK_INPUT(X);
    CHECK_INPUT(sinCosVec);
    TORCH_CHECK(X.dim() == 3, "tensor X must have 3 dims");
    TORCH_CHECK(sinCosVec.dim() == 3, "tensor sinCosVec must have 3 dims");
    TORCH_CHECK(X.size(1) == sinCosVec.size(0) && X.size(2) == sinCosVec.size(1) * 2, "tensor size should match");
    TORCH_CHECK(sinCosVec.size(2) == 2U, "Tensor sinCosVec last dimension must be equal 2");

    const unsigned batchSize = X.size(0);
    const unsigned seqLen = X.size(1);
    const unsigned dim = X.size(2);

    TORCH_CHECK(dim % 2U == 0U, "Tensor X last dimension must be even");

    auto Y = torch::empty({batchSize, seqLen, dim}, X.options());

    constexpr unsigned blockDim = 256U;
    launch_rope_cached<blockDim>(X.data_ptr<float>(), sinCosVec.data_ptr<float>(), Y.data_ptr<float>(), batchSize, seqLen, dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return Y;
}
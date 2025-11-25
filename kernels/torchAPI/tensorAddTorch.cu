#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <tensorAddTorch.hpp>

#include <vectorAdd.cuh>
#include <util.cuh>

#include "common.cuh"

torch::Tensor tensorAdd(torch::Tensor x, torch::Tensor y)
{
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y tensors must have the same size");

    const unsigned size = x.numel();
    auto z = torch::empty_like(x);
    
    const unsigned blockDim = 256;
    const unsigned gridDim = CEIL_DIV(size, blockDim);

    vectorAdd_kernel<<<gridDim, blockDim>>>(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return z;
}
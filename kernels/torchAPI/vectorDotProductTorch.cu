#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <vectorDotProductTorch.hpp>
#include <vectorDotProduct.cuh>
#include <util.cuh>

#include "common.cuh"

torch::Tensor vectorDotProduct(torch::Tensor x, torch::Tensor y)
{
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    TORCH_CHECK(x.dim() == 1 && y.dim() == 1, "x and y tensors must have dimensions equal to 1");
    TORCH_CHECK(x.size(0) == y.size(0), "x and y tensors are 1D and must have the same size");

    auto z = torch::zeros({}, x.options());
    const unsigned size = x.numel();

    unsigned maxThreadsCount = 32U;

    if(size > 2U * 256U * maxThreadsCount)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        maxThreadsCount = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
    }
    const unsigned blockDim = 256;
    // when kernel with vectorized load is used each thread loads 4 elements -> multiply by 4
    const unsigned gridDim = std::min(CEIL_DIV(size, 4 * blockDim), maxThreadsCount);

    vectorDotProduct4_kernel<<<gridDim, blockDim>>>(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return z;
}
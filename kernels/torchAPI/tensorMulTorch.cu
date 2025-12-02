#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <tensorMulTorch.hpp>
#include <vectorDotProductTorch.hpp>
#include <matrixVectorMulTorch.hpp>
#include <matmulTorch.hpp>

#include <util.cuh>

#include "common.cuh"

torch::Tensor tensorMul(torch::Tensor x, torch::Tensor y)
{
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    const auto xDim = x.dim();
    const auto yDim = y.dim();
    TORCH_CHECK(xDim > 0 && yDim > 0, "x and y tensors must have the same size");

    torch::Tensor z;

    if(xDim == 1 && yDim == 1)
    {
        z = vectorDotProduct(x, y);
    }
    else if(xDim == 2 && yDim == 1)
    {
        z = matrixVectorMul(x, y);
    }
    else if(xDim == 2 && yDim == 2)
    {
        z = matrixMul(x, y);
    }
    else
        TORCH_CHECK(false, "Not supported dimensions");

    return z;
}
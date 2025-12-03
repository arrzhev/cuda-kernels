#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <matmulTorch.hpp>
#include <matmul.cuh>
#include <util.cuh>

#include "common.cuh"

torch::Tensor matmul_(torch::Tensor x, torch::Tensor y, auto launchMatmulKernel)
{
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "x and y tensors must have dimensions equal to 2");
    TORCH_CHECK(x.size(1) == y.size(0), "x and y tensors must have the same cols and rows count");
    
    const unsigned M = x.size(0);
    const unsigned K = x.size(1);
    const unsigned N = y.size(1);
    auto z = torch::empty({M, N}, x.options());

    launchMatmulKernel(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return z;
}

torch::Tensor matrixMul(torch::Tensor x, torch::Tensor y)
{
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "x and y tensors must have dimensions equal to 2");
    TORCH_CHECK(x.size(1) == y.size(0), "x and y tensors must have the same cols and rows count");
    
    const unsigned M = x.size(0);
    const unsigned K = x.size(1);
    const unsigned N = y.size(1);
    auto z = torch::empty({M, N}, x.options());

    if(M > 512U || N > 512U)
    {
        constexpr unsigned BM = 128U;
        constexpr unsigned BN = 128U;
        constexpr unsigned BK = 8U;
        constexpr unsigned TM = 8U;
        constexpr unsigned TN = 8U;

        launch_matmul_TTiles_DBuf_vec<BM, BN, BK, TM, TN>(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), M, N, K);
    }
    else if(M > 256U || N > 256U)
    {
        constexpr unsigned BM = 64U;
        constexpr unsigned BN = 64U;
        constexpr unsigned BK = 16U;
        constexpr unsigned TM = 4U;
        constexpr unsigned TN = 4U;
        launch_matmul_TTiles_DBuf_vec<BM, BN, BK, TM, TN>(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), M, N, K);
    }
    else if(M > 128U || N > 128U)
    {
        constexpr unsigned BM = 32U;
        constexpr unsigned BN = 32U;
        constexpr unsigned BK = 32U;
        constexpr unsigned TM = 4U;
        constexpr unsigned TN = 4U;

        launch_matmul_TTiles_DBuf_vec<BM, BN, BK, TM, TN>(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), M, N, K);
    }
    else
    {
        constexpr unsigned BM = 32U;
        constexpr unsigned BN = 32U;
        constexpr unsigned BK = 32U;
        launch_matmul_BTiles_DBuf<BM,BN, BK>(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), M, N, K);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return z;
}

torch::Tensor matmulNaive(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned blockDim = 16U;

    return matmul_(x, y, launch_matmul_naive<blockDim, blockDim>);
}

torch::Tensor matmulCoalescing(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned blockDim = 16U;

    return matmul_(x, y, launch_matmul_coalescing<blockDim, blockDim>);
}

torch::Tensor matmulBTiles(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 16U;
    constexpr unsigned BN = 16U;
    constexpr unsigned BK = 16U;

    return matmul_(x, y, launch_matmul_BTiles<BM,BN, BK>);
}

torch::Tensor matmulBTilesDBuf(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 16U;
    constexpr unsigned BN = 16U;
    constexpr unsigned BK = 16U;

    return matmul_(x, y, launch_matmul_BTiles_DBuf<BM,BN, BK>);
}

torch::Tensor matmulTTiles1D(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 64U;
    constexpr unsigned BN = 64U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 16U;
    constexpr unsigned TN = 1U;

    return matmul_(x, y, launch_matmul_TTiles<BM, BN, BK, TM, TN>);
}

torch::Tensor matmulTTiles1DDBuf(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 64U;
    constexpr unsigned BN = 64U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 16U;
    constexpr unsigned TN = 1U;

    return matmul_(x, y, launch_matmul_TTiles_DBuf<BM, BN, BK, TM, TN>);
}

torch::Tensor matmulTTiles2D(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    return matmul_(x, y, launch_matmul_TTiles<BM, BN, BK, TM, TN>);
}

torch::Tensor matmulTTiles2DDBuf(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    return matmul_(x, y, launch_matmul_TTiles_DBuf<BM, BN, BK, TM, TN>);
}

torch::Tensor matmulTTiles2DVec(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    return matmul_(x, y, launch_matmul_TTiles_vec<BM, BN, BK, TM, TN>);
}

torch::Tensor matmulTTiles2DDBufVec(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    return matmul_(x, y, launch_matmul_TTiles_DBuf_vec<BM, BN, BK, TM, TN>);
}
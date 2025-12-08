#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <matmulTorch.hpp>
#include <matmul.cuh>
#include <util.cuh>

#include "common.cuh"

torch::Tensor matmul_(torch::Tensor A, torch::Tensor B, auto launchMatmulKernel)
{
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_FP32(A);
    CHECK_FP32(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "x and y tensors must have dimensions equal to 2");
    TORCH_CHECK(A.size(1) == B.size(0), "x and y tensors must have the same cols and rows count");
    
    const unsigned M = A.size(0);
    const unsigned K = A.size(1);
    const unsigned N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    bool AT = A.stride(0) == 1 ? true : false;
    bool BT = B.stride(0) == 1 ? true : false;

    launchMatmulKernel(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, AT, BT);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}

torch::Tensor matrixMul(torch::Tensor A, torch::Tensor B)
{
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_FP32(A);
    CHECK_FP32(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "x and y tensors must have dimensions equal to 2");
    TORCH_CHECK(A.size(1) == B.size(0), "x and y tensors must have the same cols and rows count");
    
    const unsigned M = A.size(0);
    const unsigned K = A.size(1);
    const unsigned N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    bool AT = A.stride(0) == 1 ? true : false;
    bool BT = B.stride(0) == 1 ? true : false;

    if(M > 512U || N > 512U)
    {
        constexpr unsigned BM = 128U;
        constexpr unsigned BN = 128U;
        constexpr unsigned BK = 8U;
        constexpr unsigned TM = 8U;
        constexpr unsigned TN = 8U;

        launch_matmul_Tiles_DBuf<BM, BN, BK, TM, TN>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, AT, BT);
    }
    else if(M > 256U || N > 256U)
    {
        constexpr unsigned BM = 64U;
        constexpr unsigned BN = 64U;
        constexpr unsigned BK = 16U;
        constexpr unsigned TM = 4U;
        constexpr unsigned TN = 4U;
        launch_matmul_Tiles_DBuf<BM, BN, BK, TM, TN>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, AT, BT);
    }
    else if(M > 128U || N > 128U)
    {
        constexpr unsigned BM = 32U;
        constexpr unsigned BN = 32U;
        constexpr unsigned BK = 32U;
        constexpr unsigned TM = 4U;
        constexpr unsigned TN = 4U;

        launch_matmul_Tiles_DBuf<BM, BN, BK, TM, TN>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, AT, BT);
    }
    else
    {
        constexpr unsigned BM = 32U;
        constexpr unsigned BN = 32U;
        constexpr unsigned BK = 32U;
        constexpr unsigned TM = 1U;
        constexpr unsigned TN = 1U;

        launch_matmul_Tiles_DBuf<BM, BN, BK, TM, TN, false>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, AT, BT);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
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
    constexpr unsigned TM = 1U;
    constexpr unsigned TN = 1U;

    return matmul_(x, y, launch_matmul_Tiles<BM, BN, BK, TM, TN, false>);
}

torch::Tensor matmulBTilesDBuf(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 16U;
    constexpr unsigned BN = 16U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 1U;
    constexpr unsigned TN = 1U;

    return matmul_(x, y, launch_matmul_Tiles_DBuf<BM, BN, BK, TM, TN, false>);
}

torch::Tensor matmulTTiles1D(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 64U;
    constexpr unsigned BN = 64U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 16U;
    constexpr unsigned TN = 1U;

    return matmul_(x, y, launch_matmul_Tiles<BM, BN, BK, TM, TN, false>);
}

torch::Tensor matmulTTiles1DDBuf(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 64U;
    constexpr unsigned BN = 64U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 16U;
    constexpr unsigned TN = 1U;

    return matmul_(x, y, launch_matmul_Tiles_DBuf<BM, BN, BK, TM, TN, false>);
}

torch::Tensor matmulTTiles2D(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    return matmul_(x, y, launch_matmul_Tiles<BM, BN, BK, TM, TN, false>);
}

torch::Tensor matmulTTiles2DDBuf(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    return matmul_(x, y, launch_matmul_Tiles_DBuf<BM, BN, BK, TM, TN, false>);
}

torch::Tensor matmulTTiles2DVec(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    return matmul_(x, y, launch_matmul_Tiles<BM, BN, BK, TM, TN>);
}

torch::Tensor matmulTTiles2DDBufVec(torch::Tensor x, torch::Tensor y)
{
    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    return matmul_(x, y, launch_matmul_Tiles_DBuf<BM, BN, BK, TM, TN>);
}
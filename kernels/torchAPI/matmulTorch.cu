#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <matmulTorch.hpp>
#include <matmul.cuh>
#include <util.cuh>

#include "common.cuh"

#define CHECK_MATMUL(A, B) CHECK_FPS(A, B); CHECK_CUDA(A); CHECK_CUDA(B); \
        TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "x and y tensors must have dimensions equal to 2"); \
        TORCH_CHECK(A.size(1) == B.size(0), "x and y tensors must have the same cols and rows count");

template <bool COAL, unsigned SK, unsigned BLOCK_DIM>
static torch::Tensor launchMatmulNaive(torch::Tensor A, torch::Tensor B)
{
    const unsigned M = A.size(0);
    const unsigned K = A.size(1);
    const unsigned N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    if constexpr(SK > 1U)
        C.zero_();

    bool AT = A.stride(0) == 1 ? true : false;
    bool BT = B.stride(0) == 1 ? true : false;

    if(A.scalar_type() == torch::kHalf)
    {
        launch_matmul_naive<BLOCK_DIM, BLOCK_DIM, COAL, SK>(
            reinterpret_cast<__half*>(A.data_ptr<torch::Half>()),
            reinterpret_cast<__half*>(B.data_ptr<torch::Half>()),
            reinterpret_cast<__half*>(C.data_ptr<torch::Half>()),
            M, N, K, AT, BT);
    }
    else
    {
        launch_matmul_naive<BLOCK_DIM, BLOCK_DIM, COAL, SK>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
             M, N, K, AT, BT);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}

template <bool DBUF, bool VEC, unsigned SK,
          unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN>
static torch::Tensor launchMatmulTiles(torch::Tensor A, torch::Tensor B)
{
    const unsigned M = A.size(0);
    const unsigned K = A.size(1);
    const unsigned N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    if constexpr(SK > 1U)
        C.zero_();

    bool AT = A.stride(0) == 1 ? true : false;
    bool BT = B.stride(0) == 1 ? true : false;

    launch_matmul_tiles<BM, BN, BK, TM, TN, VEC, DBUF, SK>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
            M, N, K, AT, BT);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}

template <bool DBUF, bool VEC, unsigned SK,
          unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN>
static torch::Tensor launchMatmulTilesHalf(torch::Tensor A, torch::Tensor B)
{   
    const unsigned M = A.size(0);
    const unsigned K = A.size(1);
    const unsigned N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    if constexpr(SK > 1U)
        C.zero_();

    bool AT = A.stride(0) == 1 ? true : false;
    bool BT = B.stride(0) == 1 ? true : false;

    launch_matmul_tiles<BM, BN, BK, TM, TN, VEC, DBUF, SK>(
        reinterpret_cast<__half*>(A.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(B.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<torch::Half>()),
        M, N, K, AT, BT);


    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}

torch::Tensor matrixMul(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);
    
    const unsigned M = A.size(0);
    const unsigned K = A.size(1);
    const unsigned N = B.size(1);

    const bool AT = A.stride(0) == 1 ? true : false;
    const bool BT = B.stride(0) == 1 ? true : false;

    const unsigned lda = AT ? M : K;
    const unsigned ldb = BT ? K : N;
    const unsigned ldc = N;
    auto possibleVecAccess = [lda, ldb, ldc](const unsigned vecSize){
        return (lda % vecSize == 0U) && (ldb % vecSize == 0U) && (ldc % vecSize == 0U);
    };

    constexpr bool DBUF = true;
    constexpr unsigned numSplitK = 1U;

    if(M > 512U || N > 512U)
    {
        constexpr unsigned BM = 128U;
        constexpr unsigned BN = 128U;
        constexpr unsigned TM = 8U;
        constexpr unsigned TN = 8U;
        
        if(A.scalar_type() == torch::kHalf)
        {
            if(possibleVecAccess(8U))
            {
                constexpr unsigned BK = 16U;
                constexpr bool VEC = true;
                return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
            }
            else
            {
                constexpr unsigned BK = 8U;
                constexpr bool VEC = false;
                return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
            }
        }
        else
        {
            constexpr unsigned BK = 8U;
            constexpr bool VEC = true;
            return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
        }
    }
    else if(M > 256U || N > 256U)
    {
        constexpr unsigned BM = 64U;
        constexpr unsigned BN = 64U;
        constexpr unsigned BK = 16U;

        if(A.scalar_type() == torch::kHalf)
        {
            if(possibleVecAccess(8U))
            {
                constexpr unsigned TM = 8U;
                constexpr unsigned TN = 8U;
                constexpr bool VEC = true;
                return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
            }
            else
            {
                constexpr unsigned TM = 4U;
                constexpr unsigned TN = 4U;
                constexpr bool VEC = false;
                return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
            }
        }
        else
        {
            constexpr unsigned TM = 4U;
            constexpr unsigned TN = 4U;
            constexpr bool VEC = false;
            return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
        }
    }
    else if(M > 128U || N > 128U)
    {
        constexpr unsigned BM = 32U;
        constexpr unsigned BN = 32U;
        constexpr unsigned BK = 32U;

        if(A.scalar_type() == torch::kHalf)
        {
            if(possibleVecAccess(8U))
            {
                constexpr unsigned TM = 8U;
                constexpr unsigned TN = 8U;
                constexpr bool VEC = true;
                return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
            }
            else
            {
                constexpr unsigned TM = 4U;
                constexpr unsigned TN = 4U;
                constexpr bool VEC = false;
                return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
            }
        }
        else
        {
            constexpr unsigned TM = 4U;
            constexpr unsigned TN = 4U;
            constexpr bool VEC = false;
            return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
        }
    }
    else
    {
        constexpr unsigned BM = 32U;
        constexpr unsigned BN = 32U;
        constexpr unsigned BK = 32U;
        constexpr unsigned TM = 1U;
        constexpr unsigned TN = 1U;

        if(A.scalar_type() == torch::kHalf)
            return launchMatmulTilesHalf<DBUF, false, numSplitK, BM, BN, BK, TM, TN>(A, B);
        else
            return launchMatmulTiles<DBUF, false, numSplitK, BM, BN, BK, TM, TN>(A, B);
    }
}

torch::Tensor matmulNaive(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr bool coalescing = false;
    constexpr unsigned numSplitK = 1U;
    constexpr unsigned blockDim = 16U;

    return launchMatmulNaive<coalescing, numSplitK, blockDim>(A, B);
}

torch::Tensor matmulNaiveK(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr bool coalescing = false;
    constexpr unsigned numSplitK = 4U;
    constexpr unsigned blockDim = 16U;

    return launchMatmulNaive<coalescing, numSplitK, blockDim>(A, B);
}

torch::Tensor matmulCoalescing(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr bool coalescing = true;
    constexpr unsigned numSplitK = 1U;
    constexpr unsigned blockDim = 16U;

    return launchMatmulNaive<coalescing, numSplitK, blockDim>(A, B);
}

torch::Tensor matmulCoalescingK(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr bool coalescing = true;
    constexpr unsigned numSplitK = 4U;
    constexpr unsigned blockDim = 16U;

    return launchMatmulNaive<coalescing, numSplitK, blockDim>(A, B);
}

torch::Tensor matmulBTiles(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr unsigned BM = 16U;
    constexpr unsigned BN = 16U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 1U;
    constexpr unsigned TN = 1U;

    constexpr bool DBUF = false;
    constexpr bool VEC = false;
    constexpr unsigned numSplitK = 1U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulBTilesK(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr unsigned BM = 16U;
    constexpr unsigned BN = 16U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 1U;
    constexpr unsigned TN = 1U;

    constexpr bool DBUF = false;
    constexpr bool VEC = false;
    constexpr unsigned numSplitK = 4U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulBTilesDBuf(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr unsigned BM = 16U;
    constexpr unsigned BN = 16U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 1U;
    constexpr unsigned TN = 1U;

    constexpr bool DBUF = true;
    constexpr bool VEC = false;
    constexpr unsigned numSplitK = 1U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulBTilesDBufK(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr unsigned BM = 16U;
    constexpr unsigned BN = 16U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 1U;
    constexpr unsigned TN = 1U;

    constexpr bool DBUF = true;
    constexpr bool VEC = false;
    constexpr unsigned numSplitK = 4U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulTTiles1D(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr unsigned BM = 64U;
    constexpr unsigned BN = 64U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 16U;
    constexpr unsigned TN = 1U;

    constexpr bool DBUF = false;
    constexpr bool VEC = false;
    constexpr unsigned numSplitK = 1U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulTTiles1DK(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr unsigned BM = 64U;
    constexpr unsigned BN = 64U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 16U;
    constexpr unsigned TN = 1U;

    constexpr bool DBUF = false;
    constexpr bool VEC = false;
    constexpr unsigned numSplitK = 4U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulTTiles1DDBuf(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr unsigned BM = 64U;
    constexpr unsigned BN = 64U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 16U;
    constexpr unsigned TN = 1U;

    constexpr bool DBUF = true;
    constexpr bool VEC = false;
    constexpr unsigned numSplitK = 1U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulTTiles1DDBufK(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr unsigned BM = 64U;
    constexpr unsigned BN = 64U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 16U;
    constexpr unsigned TN = 1U;

    constexpr bool DBUF = true;
    constexpr bool VEC = false;
    constexpr unsigned numSplitK = 4U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulTTiles2D(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    constexpr bool DBUF = false;
    constexpr bool VEC = false;
    constexpr unsigned numSplitK = 1U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulTTiles2DK(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    constexpr bool DBUF = false;
    constexpr bool VEC = false;
    constexpr unsigned numSplitK = 4U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulTTiles2DDBuf(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    constexpr bool DBUF = false;
    constexpr bool VEC = false;
    constexpr unsigned numSplitK = 1U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulTTiles2DDBufK(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    constexpr bool DBUF = false;
    constexpr bool VEC = false;
    constexpr unsigned numSplitK = 4U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulTTiles2DVec(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    constexpr bool DBUF = false;
    constexpr bool VEC = true;
    constexpr unsigned numSplitK = 1U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulTTiles2DVecK(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);

    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    constexpr bool DBUF = false;
    constexpr bool VEC = true;
    constexpr unsigned numSplitK = 4U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulTTiles2DDBufVec(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);
    
    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    constexpr bool DBUF = true;
    constexpr bool VEC = true;
    constexpr unsigned numSplitK = 1U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulTTiles2DDBufVecK(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);
    
    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 16U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;

    constexpr bool DBUF = true;
    constexpr bool VEC = true;
    constexpr unsigned numSplitK = 4U;

    if(A.scalar_type() == torch::kHalf)
        return launchMatmulTilesHalf<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);

    return launchMatmulTiles<DBUF, VEC, numSplitK, BM, BN, BK, TM, TN>(A, B);
}

torch::Tensor matmulBTilesVecWMMA(torch::Tensor A, torch::Tensor B)
{
    CHECK_MATMUL(A, B);
    CHECK_FP16(A);
    CHECK_FP16(B);

    constexpr unsigned BM = 64U;
    constexpr unsigned BN = 64U;
    constexpr unsigned BK = 64U;
    constexpr unsigned WM = 16U;
    constexpr unsigned WN = 16U;
    constexpr unsigned WK = 16U;

    constexpr bool VEC = true;

    const unsigned M = A.size(0);
    const unsigned K = A.size(1);
    const unsigned N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    bool AT = A.stride(0) == 1 ? true : false;
    bool BT = B.stride(0) == 1 ? true : false;

    launch_matmul_tiles_wmma<BM, BN, BK, WM, WN, WK, VEC>(
        reinterpret_cast<__half*>(A.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(B.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<torch::Half>()),
        M, N, K, AT, BT);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}
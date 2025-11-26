#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <matrixMulTorch.hpp>
#include <matrixMul.cuh>
#include <util.cuh>

#include "common.cuh"

static void launchMatMulNaiveKernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const dim3 blockDim(16, 16);
    const dim3 gridDim(CEIL_DIV(M, blockDim.x), CEIL_DIV(N, blockDim.y));

    matMul_naive_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
};

static void launchMatMulCoalescingKernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const dim3 blockDim(16, 16);
    const dim3 gridDim(CEIL_DIV(N, blockDim.x), CEIL_DIV(M, blockDim.y));

    matMul_coalescing_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
};

static void launchMatMulTiledKernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned tileSize = 16U;
    const dim3 blockDim(tileSize, tileSize);
    const dim3 gridDim(CEIL_DIV(N, blockDim.x), CEIL_DIV(M, blockDim.y));

    matMul_tiled_kernel<tileSize><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
};

static void launchMatMulTiled1DKernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned BM = 32U;
    constexpr unsigned BN = 32U;
    constexpr unsigned BK = 4U;
    constexpr unsigned TM = 8U;
    const dim3 blockDim(BM * BN / TM);
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    matMul_tiled_1D_kernel<BM, BN, BK, TM><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
};

static void launchMatMulTiled2DKernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;
    const dim3 blockDim(BM * BN / (TM * TN));
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    matMul_tiled_2D_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
};

static void launchMatMulTiled2D4Kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;
    const dim3 blockDim(BM * BN / (TM * TN));
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    auto launchKernel = N % 4U == 0U && K % 4U == 0U ? matMul_tiled4_2D_kernel<BM, BN, BK, TM, TN> : matMul_tiled_2D_kernel<BM, BN, BK, TM, TN>;
    launchKernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
};

torch::Tensor matrixMul_(torch::Tensor x, torch::Tensor y, auto launchMatrixMulKernel)
{
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "x and y tensors must have dimensions equal to 2");
    TORCH_CHECK(x.size(1) == y.size(0), "x and y tensors must have the same cols and rows count");
    
    const unsigned M = x.size(0);
    const unsigned K = x.size(1);
    const unsigned N = y.size(1);
    auto z = torch::empty({M, N}, x.options());

    launchMatrixMulKernel(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), M, N, K);

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

    const bool useOpt = M < 128U || N > 384U;
    if(useOpt)
        launchMatMulTiled2D4Kernel(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), M, N, K);
    else
        launchMatMulCoalescingKernel(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), M, N, K);

    return z;
}

torch::Tensor matrixMulNaive(torch::Tensor x, torch::Tensor y)
{   
    return matrixMul_(x, y, launchMatMulNaiveKernel);
}

torch::Tensor matrixMulCoalescing(torch::Tensor x, torch::Tensor y)
{
    return matrixMul_(x, y, launchMatMulCoalescingKernel);
}

torch::Tensor matrixMulTiled(torch::Tensor x, torch::Tensor y)
{
    return matrixMul_(x, y, launchMatMulTiledKernel);
}

torch::Tensor matrixMulTiled1D(torch::Tensor x, torch::Tensor y)
{
    return matrixMul_(x, y, launchMatMulTiled1DKernel);
}

torch::Tensor matrixMulTiled2D(torch::Tensor x, torch::Tensor y)
{
    return matrixMul_(x, y, launchMatMulTiled2DKernel);
}

torch::Tensor matrixMulTiled2D4(torch::Tensor x, torch::Tensor y)
{
    return matrixMul_(x, y, launchMatMulTiled2D4Kernel);
}
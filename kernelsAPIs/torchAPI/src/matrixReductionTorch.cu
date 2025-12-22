#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <matrixReductionTorch.hpp>
#include <matrixReduction.cuh>
#include <util.cuh>

#include "common.cuh"

torch::Tensor matrixRowReduction(torch::Tensor A)
{
    CHECK_FP32(A);
    CHECK_CUDA(A);
    CHECK_CONTIGUOUS(A)
    TORCH_CHECK(A.dim() == 2, "tensor must be a matrix");

    const unsigned rows = A.size(0);
    const unsigned cols = A.size(1);

    auto B = torch::empty({cols}, A.options());

    constexpr unsigned blockDim = 256U;
    // Simplified example of kernel hyper parameters selection based on input data
    const bool useOpt = rows > 2U * cols;
    if(useOpt)
        launch_matrixRowReduction<blockDim, false>(A.data_ptr<float>(), static_cast<float*>(nullptr), B.data_ptr<float>(), rows, cols);
    else
        launch_matrixRowReduction_naive<blockDim, false>(A.data_ptr<float>(), static_cast<float*>(nullptr), B.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return B;
}

torch::Tensor matrixRowReductionReLU(torch::Tensor A, torch::Tensor AR)
{
    CHECK_FP32(A);
    CHECK_CUDA(A);
    CHECK_CONTIGUOUS(A)
    TORCH_CHECK(A.dim() == 2, "tensor must be a matrix");

    const unsigned rows = A.size(0);
    const unsigned cols = A.size(1);

    auto B = torch::empty({cols}, A.options());

    constexpr unsigned blockDim = 256U;
    // Simplified example of kernel hyper parameters selection based on input data
    const bool useOpt = rows > 2U * cols;
    if(useOpt)
        launch_matrixRowReduction<blockDim, true>(A.data_ptr<float>(), AR.data_ptr<float>(), B.data_ptr<float>(), rows, cols);
    else
        launch_matrixRowReduction_naive<blockDim, true>(A.data_ptr<float>(), AR.data_ptr<float>(), B.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return B;
}
#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <util.cuh>
#include <meanBlur.cuh>
#include <meanBlurTorch.hpp>

#include "common.cuh"

torch::Tensor meanBlur(torch::Tensor image, const unsigned kernelSize)
{
    CHECK_INPUT(image);
    TORCH_CHECK(kernelSize % 2 == 1, "kernelSize must be odd");
    const bool isDim3 = image.dim() == 3;
    const bool isDim2 = image.dim() == 2;
    const bool isGray = isDim2 || (isDim3 && image.size(0) == 1);
    const bool isColored = isDim3 && image.size(0) == 3;
    TORCH_CHECK(isGray || isColored, "input tensor should be 2 dims gray image or 3 dims image in planar(CHW) format");

    if(kernelSize == 1)
        return image.clone();

    auto blurredImage = torch::empty_like(image);

    const int kernelRadius = (kernelSize - 1) / 2;
    const unsigned rows = isDim2 ? image.size(0) : image.size(1);
    const unsigned cols = isDim2 ? image.size(1) : image.size(2);

    auto src = image.data_ptr<unsigned char>();
    auto dst = blurredImage.data_ptr<unsigned char>();
    const dim3 blockDim(16, 16);
    const dim3 gridDim(CEIL_DIV(cols, blockDim.x), CEIL_DIV(rows, blockDim.y));

    if(isColored)
    {
        const unsigned size = rows * cols;
        const auto srcR = src;
        const auto srcG = srcR + size;
        const auto srcB = srcG + size;
        auto dstR = dst;
        auto dstG = dstR + size;
        auto dstB = dstG + size;

        meanBlurColor_kernel<<<gridDim, blockDim>>>(srcR, srcG, srcB, dstR, dstG, dstB, rows, cols, kernelRadius);
    }
    else
        meanBlurGray_kernel<<<gridDim, blockDim>>>(src, dst, rows, cols, kernelRadius);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return blurredImage;
}
#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <util.cuh>
#include <meanBlur.cuh>
#include <meanBlurTorch.hpp>

#include "common.cuh"

torch::Tensor meanBlur(torch::Tensor image, const unsigned kernelSize)
{
    CHECK_INPUT(image);
    CHECK_UINT8(image);
    TORCH_CHECK(kernelSize % 2 == 1, "kernelSize must be odd");
    const bool isDim3 = image.dim() == 3;
    const bool isDim2 = image.dim() == 2;
    const bool isGray = isDim2 || (isDim3 && image.size(0) == 1);
    const bool isColored = isDim3 && image.size(0) == 3;
    TORCH_CHECK(isGray || isColored, "input tensor should be 2 dims gray image or 3 dims image in planar(CHW) format");

    // No processing needed, just clone
    if(kernelSize == 1U)
        return image.clone();

    auto blurredImage = torch::empty_like(image);

    const int kernelRadius = (kernelSize - 1) / 2;
    const unsigned rows = isDim2 ? image.size(0) : image.size(1);
    const unsigned cols = isDim2 ? image.size(1) : image.size(2);

    const auto src = image.data_ptr<unsigned char>();
    auto dst = blurredImage.data_ptr<unsigned char>();

    constexpr unsigned blockDimX = 16U;
    constexpr unsigned blockDimY = 16U;
    auto launchFunction = isColored ? launch_meanBlurColor<blockDimX, blockDimY> : launch_meanBlurGray<blockDimX, blockDimY>;
    launchFunction(src, dst, rows, cols, kernelRadius);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return blurredImage;
}
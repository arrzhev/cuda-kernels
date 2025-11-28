#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <util.cuh>
#include <grayscale.cuh>
#include <grayscaleTorch.hpp>

#include "common.cuh"

torch::Tensor rgb2gray(torch::Tensor rgbImage)
{
    CHECK_CUDA(rgbImage);
    TORCH_CHECK(rgbImage.dim() == 3, "input tensor should have 3 dims");
    const bool isInterleaved = rgbImage.size(2) == 3;
    const bool isPlanar = rgbImage.size(0) == 3;
    TORCH_CHECK(isInterleaved || isPlanar, "input tensor should be an image in interleaved or planar format");

    const int64_t rows = isInterleaved ? rgbImage.size(0) : rgbImage.size(1);
    const int64_t cols = isInterleaved ? rgbImage.size(1) : rgbImage.size(2);

    auto graySize = isInterleaved ? std::vector<int64_t>{rows, cols, 1} : std::vector<int64_t>{1, rows, cols};
    auto grayImage = torch::empty(graySize, rgbImage.options());

    auto src = rgbImage.data_ptr<unsigned char>();
    auto dst = grayImage.data_ptr<unsigned char>();

    if(rgbImage.is_contiguous())
    {
        const unsigned size = rows * cols;
        constexpr unsigned blockDim = 256U;

        auto launchFunction = isPlanar ? launch_rgb2grayPlanar<blockDim> : launch_rgb2grayInterleaved<blockDim>;
        launchFunction(src, dst, size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    else
    {
        const unsigned rowStride = isInterleaved ? rgbImage.stride(0) : rgbImage.stride(1);
        const unsigned colStride = isInterleaved ? rgbImage.stride(1) : rgbImage.stride(2);
        const unsigned channelStride = isInterleaved ? rgbImage.stride(2) : rgbImage.stride(0);
        
        constexpr unsigned blockDimX = 16U;
        constexpr unsigned blockDimY = 16U;
        launch_rgb2grayStrides<blockDimX, blockDimY>(src, dst, rows, cols, rowStride, colStride, channelStride);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return grayImage;
}
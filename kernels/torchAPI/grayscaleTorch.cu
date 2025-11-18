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

    const bool isContiguous = rgbImage.is_contiguous();
    if(isContiguous)
    {
        const unsigned size = rows * cols;
        const unsigned blockDim = 256;
        const unsigned gridDim = cdiv(size, blockDim);
        
        if(isPlanar)
        {
            auto srcR = src;
            auto srcG = srcR + size;
            auto srcB = srcG + size;

            rgb2grayPlanar_kernel<<<gridDim, blockDim>>>(srcR, srcG, srcB, dst, size);
        }
        else
            rgb2grayInterleaved_kernel<<<gridDim, blockDim>>>(reinterpret_cast<const uchar3*>(src), dst, size);
    }
    else
    {
        const dim3 blockDim(16, 16);
        const dim3 gridDim(cdiv(cols, blockDim.x), cdiv(rows, blockDim.y));

        int64_t rowStride = isInterleaved ? rgbImage.stride(0) : rgbImage.stride(1);
        int64_t colStride = isInterleaved ? rgbImage.stride(1) : rgbImage.stride(2);
        int64_t channelStride = isInterleaved ? rgbImage.stride(2) : rgbImage.stride(0);

        rgb2grayStrides_kernel<<<gridDim, blockDim>>>(src, dst, rows, cols, rowStride, colStride, channelStride);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return grayImage;
}
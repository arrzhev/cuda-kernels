#ifndef MEAN_BLUR_KERNELS
#define MEAN_BLUR_KERNELS

#include <util.cuh>

__global__ void meanBlurGray_kernel(const unsigned char* src, unsigned char* dst,
     unsigned width, unsigned height, int kernelRadius);

__global__ void meanBlurColor_kernel(const unsigned char* srcR, const unsigned char* srcG, const unsigned char* srcB,
     unsigned char* dstR, unsigned char* dstG, unsigned char* dstB,
     unsigned width, unsigned height, int kernelRadius);

///// Launch functions /////

template <unsigned blockDimX, unsigned blockDimY>
void launch_meanBlurGray(const unsigned char* src, unsigned char* dst, unsigned cols, unsigned rows, int kernelRadius)
{
    const dim3 blockDim(blockDimX, blockDimY);
    const dim3 gridDim(CEIL_DIV(cols, blockDim.x), CEIL_DIV(rows, blockDim.y));
    meanBlurGray_kernel<<<gridDim, blockDim>>>(src, dst, rows, cols, kernelRadius);
}

template <unsigned blockDimX, unsigned blockDimY>
void launch_meanBlurColor(const unsigned char* src, unsigned char* dst, unsigned cols, unsigned rows, int kernelRadius)
{
    const unsigned size = rows * cols;
    const auto srcR = src;
    const auto srcG = srcR + size;
    const auto srcB = srcG + size;
    auto dstR = dst;
    auto dstG = dstR + size;
    auto dstB = dstG + size;

    const dim3 blockDim(blockDimX, blockDimY);
    const dim3 gridDim(CEIL_DIV(cols, blockDim.x), CEIL_DIV(rows, blockDim.y));
    meanBlurColor_kernel<<<gridDim, blockDim>>>(srcR, srcG, srcB, dstR, dstG, dstB, rows, cols, kernelRadius);
}

#endif // MEAN_BLUR_KERNELS
#ifndef GRAYSCALE_KERNELS
#define GRAYSCALE_KERNELS

#include <concepts>
#include <util.cuh>

template <typename T>
concept ucharDim = IsAnyOf<T, unsigned char, uchar4>;

__device__ __forceinline__ unsigned char rgb2gray(unsigned char r, unsigned char g, unsigned char b)
{
    return static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
}

__device__ __forceinline__ uchar4 rgb2gray(uchar4 r, uchar4 g, uchar4 b)
{
    uchar4 result;
    result.x = rgb2gray(r.x, g.x, b.x);
    result.y = rgb2gray(r.y, g.y, b.y);
    result.z = rgb2gray(r.z, g.z, b.z);
    result.w = rgb2gray(r.w, g.w, b.w);
    return result;
}

__global__ void rgb2grayInterleaved_kernel(const uchar3* src, unsigned char* dst, unsigned size);

__global__ void rgb2grayStrides_kernel(const unsigned char* src, unsigned char* dst, unsigned rows, unsigned cols,
     unsigned rowStride, unsigned colStride, unsigned channelStride);

template <ucharDim T>
__global__ void rgb2grayPlanar_kernel(const T* srcR, const T* srcG, const T* srcB, T* dst, unsigned size)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < size)
        dst[i] = rgb2gray(srcR[i], srcG[i], srcB[i]);
}

///// Launch functions /////

template <unsigned blockDim>
void launch_rgb2grayInterleaved(const unsigned char* src, unsigned char* dst, unsigned size)
{
     const unsigned gridDim = CEIL_DIV(size, blockDim);
     rgb2grayInterleaved_kernel<<<gridDim, blockDim>>>(reinterpret_cast<const uchar3*>(src), dst, size);
}

template <unsigned blockDim>
void launch_rgb2grayPlanar(const unsigned char* src, unsigned char* dst, unsigned size)
{
     if(size % 4U == 0U)
     {
          const unsigned size4 = size / 4U;

          auto srcR = reinterpret_cast<const uchar4*>(src);
          auto srcG = srcR + size4;
          auto srcB = srcG + size4;

          const unsigned gridDim = CEIL_DIV(size4, blockDim);
          rgb2grayPlanar_kernel<<<gridDim, blockDim>>>(srcR, srcG, srcB, reinterpret_cast<uchar4*>(dst), size4);
     }
     else
     {
          auto srcR = src;
          auto srcG = srcR + size;
          auto srcB = srcG + size;

          const unsigned gridDim = CEIL_DIV(size, blockDim);
          rgb2grayPlanar_kernel<<<gridDim, blockDim>>>(srcR, srcG, srcB, dst, size);
     }
}

template <unsigned blockDimX, unsigned blockDimY>
void launch_rgb2grayStrides(const unsigned char* src, unsigned char* dst, unsigned rows, unsigned cols,
     unsigned rowStride, unsigned colStride, unsigned channelStride)
{
     const dim3 blockDim(blockDimX, blockDimY);
     const dim3 gridDim(CEIL_DIV(cols, blockDim.x), CEIL_DIV(rows, blockDim.y));
     rgb2grayStrides_kernel<<<gridDim, blockDim>>>(src, dst, rows, cols, rowStride, colStride, channelStride);
}

#endif // GRAYSCALE_KERNELS
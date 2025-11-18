#include <grayscale.cuh>
#include <util.cuh>

inline __device__ unsigned char rgb2gray_kernel(unsigned char r, unsigned char g, unsigned char b)
{
    return static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
}

__global__ void rgb2grayInterleaved_kernel(const uchar3* src, unsigned char* dst, unsigned size)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    auto rgb = src[i];

    if(i < size)
        dst[i] = rgb2gray_kernel(rgb.x, rgb.y, rgb.z);
}

__global__ void rgb2grayPlanar_kernel(const unsigned char* srcR, const unsigned char* srcG, const unsigned char* srcB,
     unsigned char* dst, unsigned size)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < size)
        dst[i] = rgb2gray_kernel(srcR[i], srcG[i], srcB[i]);
}


__global__ void rgb2grayStrides_kernel(const unsigned char* src, unsigned char* dst, unsigned rows, unsigned cols,
     unsigned rowStride, unsigned colStride, unsigned channelStride)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < cols && row < rows)
    {
        const unsigned iR = row * rowStride + col * colStride;
        const unsigned iG = iR + channelStride;
        const unsigned iB = iG + channelStride;

        dst[row * cols + col] = rgb2gray_kernel(src[iR], src[iG], src[iB]);
    }
}
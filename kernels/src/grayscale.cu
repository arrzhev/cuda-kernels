#include <grayscale.cuh>

__global__ void rgb2grayInterleaved_kernel(const uchar3* src, unsigned char* dst, unsigned size)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    auto rgb = src[i];

    if(i < size)
        dst[i] = rgb2gray(rgb.x, rgb.y, rgb.z);
}

__global__ void rgb2grayStrides_kernel(const unsigned char* src, unsigned char* dst, unsigned rows, unsigned cols,
     unsigned rowStride, unsigned colStride, unsigned channelStride)
{
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < cols && row < rows)
    {
        const unsigned iR = row * rowStride + col * colStride;
        const unsigned iG = iR + channelStride;
        const unsigned iB = iG + channelStride;

        dst[row * cols + col] = rgb2gray(src[iR], src[iG], src[iB]);
    }
}
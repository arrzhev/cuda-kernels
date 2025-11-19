#ifndef GRAYSCALE_KERNELS
#define GRAYSCALE_KERNELS

__global__ void rgb2grayInterleaved_kernel(const uchar3* src, unsigned char* dst, unsigned size);
__global__ void rgb2grayPlanar_kernel(const unsigned char* srcR, const unsigned char* srcG, const unsigned char* srcB, unsigned char* dst, unsigned size);
__global__ void rgb2grayStrides_kernel(const unsigned char* src, unsigned char* dst, unsigned rows, unsigned cols,
     unsigned rowStride, unsigned colStride, unsigned channelStride);

#endif // GRAYSCALE_KERNELS
#ifndef MEAN_BLUR_KERNELS
#define MEAN_BLUR_KERNELS

__global__ void meanBlurGray_kernel(const unsigned char* src, unsigned char* dst,
     unsigned width, unsigned height, int kernelRadius);

__global__ void meanBlurColor_kernel(const unsigned char* srcR, const unsigned char* srcG, const unsigned char* srcB,
     unsigned char* dstR, unsigned char* dstG, unsigned char* dstB,
     unsigned width, unsigned height, int kernelRadius);

#endif // MEAN_BLUR_KERNELS
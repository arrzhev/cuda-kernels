#include <util.cuh>
#include <grayscale.cuh>
#include <grayscaleC.hpp>

void rgb2gray(const unsigned char* src_h, unsigned char* dst_h, unsigned size, bool isPlanar)
{
    unsigned char* src_d;
    unsigned char* dst_d;
    const unsigned dstByteSize = size * sizeof(unsigned char);
    const unsigned srcByteSize = 3 * dstByteSize;

    cudaCheckErrors(cudaMalloc(&src_d, srcByteSize));
    cudaCheckErrors(cudaMalloc(&dst_d, dstByteSize));

    cudaCheckErrors(cudaMemcpy(src_d, src_h, srcByteSize, cudaMemcpyHostToDevice));

    const unsigned blockDim = 256;
    const unsigned gridDim = CEIL_DIV(size, blockDim);
    
    if(isPlanar)
    {
        auto srcR_d = src_d;
        auto srcG_d = srcR_d + size;
        auto srcB_d = srcG_d + size;

        rgb2grayPlanar_kernel<<<gridDim, blockDim>>>(srcR_d, srcG_d, srcB_d, dst_d, size);
    }
    else
        rgb2grayInterleaved_kernel<<<gridDim, blockDim>>>(reinterpret_cast<const uchar3*>(src_d), dst_d, size);

    cudaCheckErrors(cudaPeekAtLastError());
    cudaCheckErrors(cudaDeviceSynchronize());

    cudaCheckErrors(cudaMemcpy(dst_h, dst_d, dstByteSize, cudaMemcpyDeviceToHost));

    cudaCheckErrors(cudaFree(src_d));
    cudaCheckErrors(cudaFree(dst_d));
}

void rgb2grayInterleaved(const unsigned char* src, unsigned char* dst, unsigned size)
{
    return rgb2gray(src, dst, size, false);
}

void rgb2grayPlanar(const unsigned char* src, unsigned char* dst, unsigned size)
{
    return rgb2gray(src, dst, size, true);
}
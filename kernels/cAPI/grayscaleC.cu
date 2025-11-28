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

    constexpr unsigned blockDim = 256U;
    auto launchFunction = isPlanar ? launch_rgb2grayPlanar<blockDim> : launch_rgb2grayInterleaved<blockDim>;
    launchFunction(src_d, dst_d, size);

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
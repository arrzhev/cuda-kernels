#include <util.cuh>
#include <meanBlur.cuh>
#include <meanBlurC.hpp>

void meanBlur(const unsigned char* src_h, unsigned char* dst_h, unsigned rows, unsigned cols, int kernelSize, bool isColored)
{
    unsigned char* src_d;
    unsigned char* dst_d;

    const unsigned channels = isColored ? 3 : 1;
    const unsigned size = rows * cols;
    const unsigned byteSize = size * channels * sizeof(unsigned char);
    const int kernelRadius = (kernelSize - 1) / 2;

    cudaCheckErrors(cudaMalloc(&src_d, byteSize));
    cudaCheckErrors(cudaMalloc(&dst_d, byteSize));

    cudaCheckErrors(cudaMemcpy(src_d, src_h, byteSize, cudaMemcpyHostToDevice));

    constexpr unsigned blockDimX = 16U;
    constexpr unsigned blockDimY = 16U;
    auto launchFunction = isColored ? launch_meanBlurColor<blockDimX, blockDimY> : launch_meanBlurGray<blockDimX, blockDimY>;
    launchFunction(src_d, dst_d, rows, cols, kernelRadius);

    cudaCheckErrors(cudaPeekAtLastError());
    cudaCheckErrors(cudaDeviceSynchronize());

    cudaCheckErrors(cudaMemcpy(dst_h, dst_d, byteSize, cudaMemcpyDeviceToHost));

    cudaCheckErrors(cudaFree(src_d));
    cudaCheckErrors(cudaFree(dst_d));
}

void meanBlurColor(const unsigned char* src, unsigned char* dst, unsigned rows, unsigned cols, int kernelSize)
{
    return meanBlur(src, dst, rows, cols, kernelSize, true);
}

void meanBlurGray(const unsigned char* src, unsigned char* dst, unsigned rows, unsigned cols, int kernelSize)
{
    return meanBlur(src, dst, rows, cols, kernelSize, false);
}
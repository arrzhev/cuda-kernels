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

    const dim3 blockDim(16, 16);
    const dim3 gridDim(CEIL_DIV(cols, blockDim.x), CEIL_DIV(rows, blockDim.y));

    if(isColored)
    {
        const auto srcR_d = src_d;
        const auto srcG_d = srcR_d + size;
        const auto srcB_d = srcG_d + size;
        auto dstR_d = dst_d;
        auto dstG_d = dstR_d + size;
        auto dstB_d = dstG_d + size;

        meanBlurColor_kernel<<<gridDim, blockDim>>>(srcR_d, srcG_d, srcB_d, dstR_d, dstG_d, dstB_d, rows, cols, kernelRadius);
    }
    else
        meanBlurGray_kernel<<<gridDim, blockDim>>>(src_d, dst_d, rows, cols, kernelRadius);

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
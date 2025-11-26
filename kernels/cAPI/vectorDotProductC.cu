#include <vectorDotProductC.hpp>

#include <util.cuh>
#include <vectorDotProduct.cuh>

void vectorDotProduct(const float *x_h, const float *y_h, float *z_h, unsigned size)
{
    float* x_d;
    float* y_d;
    float* z_d;
    const unsigned byteSize = size * sizeof(float);

    cudaCheckErrors(cudaMalloc(&x_d, byteSize));
    cudaCheckErrors(cudaMalloc(&y_d, byteSize));
    cudaCheckErrors(cudaMalloc(&z_d, sizeof(float)));

    cudaCheckErrors(cudaMemcpy(x_d, x_h, byteSize, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(y_d, y_h, byteSize, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemset(z_d, 0, sizeof(float)));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const unsigned maxThreadsCount = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;

    const unsigned blockDim = 256;
    // when kernel with vectorized load is used each thread loads 4 elements -> multiply by 4
    const unsigned gridDim = std::min(CEIL_DIV(size, 4 * blockDim), maxThreadsCount);

    vectorDotProduct4_kernel<<<gridDim, blockDim>>>(x_d, y_d, z_d, size);
    cudaCheckErrors(cudaPeekAtLastError());
    cudaCheckErrors(cudaDeviceSynchronize());

    cudaCheckErrors(cudaMemcpy(z_h, z_d, sizeof(float), cudaMemcpyDeviceToHost));

    cudaCheckErrors(cudaFree(x_d));
    cudaCheckErrors(cudaFree(y_d));
    cudaCheckErrors(cudaFree(z_d));
}
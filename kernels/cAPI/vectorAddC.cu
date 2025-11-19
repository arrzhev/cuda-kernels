#include <vectorAddC.hpp>
#include <util.cuh>
#include <vectorAdd.cuh>

void vectorAdd(const float *x_h, const float *y_h, float *z_h, unsigned size)
{
    float* x_d;
    float* y_d;
    float* z_d;
    const unsigned byteSize = size * sizeof(float);

    cudaCheckErrors(cudaMalloc(&x_d, byteSize));
    cudaCheckErrors(cudaMalloc(&y_d, byteSize));
    cudaCheckErrors(cudaMalloc(&z_d, byteSize));

    cudaCheckErrors(cudaMemcpy(x_d, x_h, byteSize, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(y_d, y_h, byteSize, cudaMemcpyHostToDevice));

    const unsigned blockDim = 256;
    const unsigned gridDim = CEIL_DIV(size, blockDim);

    vectorAdd_kernel<<<gridDim, blockDim>>>(x_d, y_d, z_d, size);
    cudaCheckErrors(cudaPeekAtLastError());
    cudaCheckErrors(cudaDeviceSynchronize());

    cudaCheckErrors(cudaMemcpy(z_h, z_d, byteSize, cudaMemcpyDeviceToHost));

    cudaCheckErrors(cudaFree(x_d));
    cudaCheckErrors(cudaFree(y_d));
    cudaCheckErrors(cudaFree(z_d));
}
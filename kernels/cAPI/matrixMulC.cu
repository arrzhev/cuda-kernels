#include <util.cuh>
#include <matrixMul.cuh>
#include <matrixMulC.hpp>

void matrixVectorMul(const float *X_h, const float *y_h, float *z_h, unsigned rows, unsigned cols)
{
    float* X_d;
    float* y_d;
    float* z_d;
    const unsigned byteSizeX = rows * cols * sizeof(float);
    const unsigned byteSizeY = cols * sizeof(float);
    const unsigned byteSizeZ = rows * sizeof(float);

    cudaCheckErrors(cudaMalloc(&X_d, byteSizeX));
    cudaCheckErrors(cudaMalloc(&y_d, byteSizeY));
    cudaCheckErrors(cudaMalloc(&z_d, byteSizeZ));

    cudaCheckErrors(cudaMemcpy(X_d, X_h, byteSizeX, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(y_d, y_h, byteSizeY, cudaMemcpyHostToDevice));

    const unsigned blockDim = 256;
    const unsigned gridDim = cdiv(rows, blockDim);

    matrixVectorMul_naive_kernel<<<gridDim, blockDim>>>(X_d, y_d, z_d, rows, cols);
    cudaCheckErrors(cudaPeekAtLastError());
    cudaCheckErrors(cudaDeviceSynchronize());

    cudaCheckErrors(cudaMemcpy(z_h, z_d, byteSizeZ, cudaMemcpyDeviceToHost));

    cudaCheckErrors(cudaFree(X_d));
    cudaCheckErrors(cudaFree(y_d));
    cudaCheckErrors(cudaFree(z_d));
}
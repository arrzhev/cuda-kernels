#include <matrixVectorMulC.hpp>

#include <util.cuh>
#include <matrixVectorMul.cuh>

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

    constexpr unsigned blockDim = 256U;
    // Simplified example of kernel hyper parameters selection based on input data
    const bool useOpt = rows < 128U || cols > 384U;
    auto launchKernel = useOpt ? launch_matrixVectorMul_warp<blockDim> : launch_matrixVectorMul_naive<blockDim>;
    launchKernel(X_d, y_d, z_d, rows, cols);

    cudaCheckErrors(cudaPeekAtLastError());
    cudaCheckErrors(cudaDeviceSynchronize());

    cudaCheckErrors(cudaMemcpy(z_h, z_d, byteSizeZ, cudaMemcpyDeviceToHost));

    cudaCheckErrors(cudaFree(X_d));
    cudaCheckErrors(cudaFree(y_d));
    cudaCheckErrors(cudaFree(z_d));
}
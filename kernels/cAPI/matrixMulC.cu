#include <matrixMulC.hpp>

#include <util.cuh>
#include <matrixMul.cuh>

void matrixMul(const float *A_h, const float *B_h, float *C_h, unsigned M, unsigned N, unsigned K)
{
    float* A_d;
    float* B_d;
    float* C_d;
    const unsigned byteSizeA = M * K * sizeof(float);
    const unsigned byteSizeB = K * N * sizeof(float);
    const unsigned byteSizeC = M * N * sizeof(float);

    cudaCheckErrors(cudaMalloc(&A_d, byteSizeA));
    cudaCheckErrors(cudaMalloc(&B_d, byteSizeB));
    cudaCheckErrors(cudaMalloc(&C_d, byteSizeC));

    cudaCheckErrors(cudaMemcpy(A_d, A_h, byteSizeA, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(B_d, B_h, byteSizeB, cudaMemcpyHostToDevice));

    const bool useOpt = M > 128U && N > 128U;
    if(useOpt)
        launch_matMul_tiled_2D(A_d, B_d, C_d, M, N, K);
    else
        launch_matMul_coalescing(A_d, B_d, C_d, M, N, K);

    cudaCheckErrors(cudaPeekAtLastError());
    cudaCheckErrors(cudaDeviceSynchronize());

    cudaCheckErrors(cudaMemcpy(C_h, C_d, byteSizeC, cudaMemcpyDeviceToHost));

    cudaCheckErrors(cudaFree(A_d));
    cudaCheckErrors(cudaFree(B_d));
    cudaCheckErrors(cudaFree(C_d));
}
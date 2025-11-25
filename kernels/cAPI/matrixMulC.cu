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

    constexpr unsigned BM = 128U;
    constexpr unsigned BN = 128U;
    constexpr unsigned BK = 8U;
    constexpr unsigned TM = 8U;
    constexpr unsigned TN = 8U;
    const dim3 blockDim(BM * BN / (TM * TN));
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    matMul_tiled_2D_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A_d, B_d, C_d, M, N, K);
    cudaCheckErrors(cudaPeekAtLastError());
    cudaCheckErrors(cudaDeviceSynchronize());

    cudaCheckErrors(cudaMemcpy(C_h, C_d, byteSizeC, cudaMemcpyDeviceToHost));

    cudaCheckErrors(cudaFree(A_d));
    cudaCheckErrors(cudaFree(B_d));
    cudaCheckErrors(cudaFree(C_d));
}
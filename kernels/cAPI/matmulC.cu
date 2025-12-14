#include <matmulC.hpp>

#include <util.cuh>
#include <matmul.cuh>

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

    bool AT = false;
    bool BT = false;

    if(M > 512U || N > 512U)
    {
        constexpr unsigned BM = 128U;
        constexpr unsigned BN = 128U;
        constexpr unsigned BK = 8U;
        constexpr unsigned TM = 8U;
        constexpr unsigned TN = 8U;

        launch_matmul_tiles<BM, BN, BK, TM, TN>(A_d, B_d, C_d, M, N, K, AT, BT);
    }
    else if(M > 256U || N > 256U)
    {
        constexpr unsigned BM = 64U;
        constexpr unsigned BN = 64U;
        constexpr unsigned BK = 16U;
        constexpr unsigned TM = 4U;
        constexpr unsigned TN = 4U;

        launch_matmul_tiles<BM, BN, BK, TM, TN>(A_d, B_d, C_d, M, N, K, AT, BT);
    }
    else if(M > 128U || N > 128U)
    {
        constexpr unsigned BM = 32U;
        constexpr unsigned BN = 32U;
        constexpr unsigned BK = 32U;
        constexpr unsigned TM = 4U;
        constexpr unsigned TN = 4U;

        launch_matmul_tiles<BM, BN, BK, TM, TN>(A_d, B_d, C_d, M, N, K, AT, BT);
    }
    else
    {
        constexpr unsigned BM = 32U;
        constexpr unsigned BN = 32U;
        constexpr unsigned BK = 32U;
        constexpr unsigned TM = 1U;
        constexpr unsigned TN = 1U;
        constexpr bool VEC = false;

        launch_matmul_tiles<BM, BN, BK, TM, TN, VEC>(A_d, B_d, C_d, M, N, K, AT, BT);
    }

    cudaCheckErrors(cudaPeekAtLastError());
    cudaCheckErrors(cudaDeviceSynchronize());

    cudaCheckErrors(cudaMemcpy(C_h, C_d, byteSizeC, cudaMemcpyDeviceToHost));

    cudaCheckErrors(cudaFree(A_d));
    cudaCheckErrors(cudaFree(B_d));
    cudaCheckErrors(cudaFree(C_d));
}
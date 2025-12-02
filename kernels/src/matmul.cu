#include <matmul.cuh>
#include <util.cuh>

__global__ void matmul_naive_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
     {
        float sum = 0.0f;
        for (unsigned i = 0; i < K; ++i)
            sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void matmul_coalescing_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
     {
        float sum = 0.0f;
        for (unsigned i = 0; i < K; ++i)
            sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}
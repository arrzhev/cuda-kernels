#ifndef MATRIX_MUL_KERNELS
#define MATRIX_MUL_KERNELS

#include <util.cuh>

__global__ void matMul_naive_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K);
__global__ void matMul_coalescing_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K);

template <const unsigned TILE_SIZE>
__global__ void matMul_tiled_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const unsigned ty = threadIdx.y;
    const unsigned tx = threadIdx.x;

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

    const unsigned row = by * TILE_SIZE + ty;
    const unsigned col = bx * TILE_SIZE + tx;

    __shared__ float smemA[TILE_SIZE][TILE_SIZE];
    __shared__ float smemB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (unsigned tileOffset = 0; tileOffset < K; tileOffset+=TILE_SIZE)
    {
        if (row < M && (tileOffset + tx) < K)
            smemA[ty][tx] = A[row * K + tileOffset + tx];
        else
            smemA[ty][tx] = 0.0f;

        if ((tileOffset + ty) < K && col < N)
            smemB[ty][tx] = B[(tileOffset + ty) * N + col];
        else
            smemB[ty][tx] = 0.0f;
        __syncthreads();

        for (unsigned i = 0; i < TILE_SIZE; ++i)
            sum += smemA[ty][i] * smemB[i][tx];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// BM=BN=BK*TM
template <const unsigned BM, const unsigned BN, const unsigned BK, const unsigned TM>
__global__ void matMul_tiled_1D_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const unsigned ty = threadIdx.x / BN;
    const unsigned tx = threadIdx.x % BN;

    const unsigned aty = threadIdx.x / BK;
    const unsigned atx = threadIdx.x % BK;

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

    unsigned row = by * BM + ty * TM;
    unsigned col = bx * BN + tx;

    __shared__ float smemA[BM][BK];
    __shared__ float smemB[BK][BN];

    float sums[TM] = {0.0f};

    for (unsigned tileOffset = 0; tileOffset < K; tileOffset += BK)
    {
        if ((by * BM + aty) < M && (tileOffset + atx) < K)
            smemA[aty][atx] = A[(by * BM + aty) * K + (tileOffset + atx)];
        else
            smemA[aty][atx] = 0.0f;

        if ((tileOffset + ty) < K && (bx * BN + tx) < N)
            smemB[ty][tx] = B[(tileOffset + ty) * N + (bx * BN + tx)];
        else
            smemB[ty][tx] = 0.0f;

        __syncthreads();

        for (unsigned i = 0; i < BK; ++i)
        {
            float tmp = smemB[i][tx];
            for (unsigned c = 0; c < TM; ++c)
                sums[c] += smemA[ty * TM + c][i] * tmp;
        }
        __syncthreads();
    }

    for (unsigned c = 0; c < TM; ++c)
    {
        if ((row + c) < M && col < N)
            C[(row + c) * N + col] = sums[c];
    }
}

template <const unsigned BM, const unsigned BN, const unsigned BK, const unsigned TM, const unsigned TN>
__global__ void matMul_tiled_2D_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const unsigned ty = threadIdx.x / (BN / TN);
    const unsigned tx = threadIdx.x % (BN / TN);

    const unsigned aty = threadIdx.x / BK;
    const unsigned atx = threadIdx.x % BK;

    const unsigned bty = threadIdx.x / BN;
    const unsigned btx = threadIdx.x % BN;

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

    const unsigned strideA = blockDim.x / BK;
    const unsigned strideB = blockDim.x / BN;

    unsigned row = by * BM + ty * TM;
    unsigned col = bx * BN + tx * TN;

    __shared__ float smemA[BM][BK];
    __shared__ float smemB[BK][BN];

    float sums[TM][TN] = {0.0f};

    for (unsigned tileOffset = 0; tileOffset < K; tileOffset += BK)
    {
        for (unsigned loadOffset = 0; loadOffset < BM; loadOffset += strideA)
        {
            if((by * BM + (aty + loadOffset)) < M && (tileOffset + atx) < K)
                smemA[aty + loadOffset][atx] = A[(by * BM + (aty + loadOffset)) * K + (tileOffset + atx)];
            else
                smemA[aty + loadOffset][atx] = 0.0f;
        }

        for (unsigned loadOffset = 0; loadOffset < BK; loadOffset += strideB)
        {
            if((tileOffset + bty + loadOffset) < K && (bx * BN + btx) < N)
                smemB[bty + loadOffset][btx] = B[(tileOffset + bty + loadOffset) * N + (bx * BN + btx)];
            else
                smemB[bty + loadOffset][btx] = 0.0f;
        }

        __syncthreads();

        for (unsigned i = 0; i < BK; ++i)
        {
            for (unsigned cM = 0; cM < TM; ++cM)
            {
                const float tmpSumA = smemA[ty * TM + cM][i];
                for (unsigned cN = 0; cN < TN; ++cN)
                    sums[cM][cN] += tmpSumA * smemB[i][tx * TN + cN];
            }
        }

        __syncthreads();
    }

    for (unsigned cM = 0; cM < TM; ++cM)
    {
        for (unsigned cN = 0; cN < TN; ++cN)
        {
            if((row + cM) < M && col + cN < N)
                C[(row + cM) * N + col + cN] = sums[cM][cN];
        }
    }
}

#endif // MATRIX_MUL_KERNELS
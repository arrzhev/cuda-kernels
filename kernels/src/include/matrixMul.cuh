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

    for (unsigned tileOffset = 0U; tileOffset < K; tileOffset+=TILE_SIZE)
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

        for (unsigned i = 0U; i < TILE_SIZE; ++i)
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

    const unsigned row = by * BM + ty * TM;
    const unsigned col = bx * BN + tx;

    __shared__ float smemA[BM][BK];
    __shared__ float smemB[BK][BN];

    float sums[TM] = {0.0f};

    for (unsigned tileOffset = 0U; tileOffset < K; tileOffset += BK)
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

        for (unsigned i = 0U; i < BK; ++i)
        {
            float tmp = smemB[i][tx];
            for (unsigned c = 0U; c < TM; ++c)
                sums[c] += smemA[ty * TM + c][i] * tmp;
        }
        __syncthreads();
    }

    for (unsigned m = 0U; m < TM; ++m)
    {
        if ((row + m) < M && col < N)
            C[(row + m) * N + col] = sums[m];
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

    const unsigned row = by * BM + ty * TM;
    const unsigned col = bx * BN + tx * TN;

    __shared__ float smemA[BM][BK];
    __shared__ float smemB[BK][BN];

    float sums[TM][TN] = {0.0f};
    
    for (unsigned tileOffset = 0U; tileOffset < K; tileOffset += BK)
    {
        for (unsigned loadOffset = 0U; loadOffset < BM; loadOffset += strideA)
        {
            if((by * BM + (aty + loadOffset)) < M && (tileOffset + atx) < K)
                smemA[aty + loadOffset][atx] = A[(by * BM + (aty + loadOffset)) * K + (tileOffset + atx)];
            else
                smemA[aty + loadOffset][atx] = 0.0f;
        }

        for (unsigned loadOffset = 0U; loadOffset < BK; loadOffset += strideB)
        {
            if((tileOffset + bty + loadOffset) < K && (bx * BN + btx) < N)
                smemB[bty + loadOffset][btx] = B[(tileOffset + bty + loadOffset) * N + (bx * BN + btx)];
            else
                smemB[bty + loadOffset][btx] = 0.0f;
        }

        __syncthreads();

        for (unsigned i = 0U; i < BK; ++i)
        {
            for (unsigned m = 0U; m < TM; ++m)
            {
                const float tmpSumA = smemA[ty * TM + m][i];
                for (unsigned n = 0U; n < TN; ++n)
                    sums[m][n] += tmpSumA * smemB[i][tx * TN + n];
            }
        }

        __syncthreads();
    }

    for (unsigned m = 0U; m < TM; ++m)
    {
        for (unsigned n = 0U; n < TN; ++n)
        {
            if((row + m) < M && col + n < N)
                C[(row + m) * N + col + n] = sums[m][n];
        }
    }
}
// K % 4 == 0 && BK % 4 == 0 && BN % 4 == 0 && N % 4 == 0 && TM % 4 == 0 && TN % 4 == 0
template <const unsigned BM, const unsigned BN, const unsigned BK, const unsigned TM, const unsigned TN>
__global__ void matMul_tiled4_2D_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const unsigned ty = threadIdx.x / (BN / TN);
    const unsigned tx = threadIdx.x % (BN / TN);

    const unsigned aty = threadIdx.x / (BK / 4U);
    const unsigned atx = threadIdx.x % (BK / 4U);

    const unsigned bty = threadIdx.x / (BN / 4U);
    const unsigned btx = threadIdx.x % (BN / 4U);

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

    const unsigned strideA = 4U * blockDim.x / BK;
    const unsigned strideB = 4U * blockDim.x / BN;

    const unsigned row = by * BM + ty * TM;
    const unsigned col = bx * BN + tx * TN;

    __shared__ float smemA[BK][BM];
    __shared__ float smemB[BK][BN];

    float sums[TM][TN] = {0.0f};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (unsigned tileOffset = 0U; tileOffset < K; tileOffset += BK)
    {
        for (unsigned loadOffset = 0U; loadOffset < BM; loadOffset += strideA)
        {
            float4 tmpA = {0.0f};
            if((by * BM + (aty + loadOffset)) < M && (tileOffset + 4U * atx) < K)
                tmpA = reinterpret_cast<const float4*>(&A[(by * BM + (aty + loadOffset)) * K + (tileOffset + 4U * atx)])[0U];
            smemA[4U * atx][aty + loadOffset] = tmpA.x;
            smemA[4U * atx + 1U][aty + loadOffset] = tmpA.y;
            smemA[4U * atx + 2U][aty + loadOffset] = tmpA.z;
            smemA[4U * atx + 3U][aty + loadOffset] = tmpA.w;
        }

        for (unsigned loadOffset = 0U; loadOffset < BK; loadOffset += strideB)
        {
            float4 tmpB = {0.0f};
            if((tileOffset + bty + loadOffset) < K && (bx * BN + 4U * btx) < N)
                tmpB = reinterpret_cast<const float4*>(&B[(tileOffset + bty + loadOffset) * N + (bx * BN + 4U * btx)])[0U];
            reinterpret_cast<float4*>(&smemB[bty + loadOffset][4U * btx])[0U] = tmpB;
        }

        __syncthreads();

        for (unsigned i = 0U; i < BK; ++i)
        {
            for (unsigned m = 0U; m < TM; m+=4U)
                reinterpret_cast<float4*>(&regM[m])[0U] = reinterpret_cast<float4*>(&smemA[i][ty * TM + m])[0U];

            for (unsigned n = 0U; n < TN; n+=4U)
                reinterpret_cast<float4*>(&regN[n])[0U] = reinterpret_cast<float4*>(&smemB[i][tx * TN + n])[0U];

            for (unsigned m = 0U; m < TM; ++m)
            {
                for (unsigned n = 0; n < TN; ++n)
                    sums[m][n] += regM[m] * regN[n];
            }
        }

        __syncthreads();
    }

    for (unsigned m = 0U; m < TM; ++m)
    {
        for (unsigned n = 0U; n < TN; n+=4U)
        {
            if((row + m) < M && col + n < N)
                reinterpret_cast<float4*>(&C[(row + m) * N + col + n])[0U] = reinterpret_cast<float4*>(&sums[m][n])[0U];
        }
    }
}

#endif // MATRIX_MUL_KERNELS
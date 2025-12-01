#ifndef MATRIX_MUL_KERNELS
#define MATRIX_MUL_KERNELS

#include <util.cuh>

__global__ void matMul_naive_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K);
__global__ void matMul_coalescing_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K);

// BN * BK % (BM * BN) == 0U && BM * BK % (BM * BN) == 0U
template <const unsigned BM, const unsigned BN, const unsigned BK>
__global__ void matMul_tiled_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned NUM_THREADS = BM * BN;

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;
    
    const unsigned ty = threadIdx.x / BN;
    const unsigned tx = threadIdx.x % BN;

    const unsigned row = by * BM + ty;
    const unsigned col = bx * BN + tx;

    __shared__ float smemA[BM][BK];
    __shared__ float smemB[BK][BN];

    float sum = 0.0f;

    for (unsigned blockOffset = 0U; blockOffset < K; blockOffset += BK)
    {
#pragma unroll
        for (unsigned threadOffset = 0U; threadOffset < BM * BK; threadOffset += NUM_THREADS)
        {
            const unsigned rowTileA = (threadIdx.x + threadOffset) / BK;
            const unsigned colTileA = (threadIdx.x + threadOffset) % BK;
            const unsigned rowA = by * BM + rowTileA;
            const unsigned colA = blockOffset + colTileA;
            
            if(rowA < M && colA < K)
                smemA[rowTileA][colTileA] = A[rowA * K + colA];
            else
                smemA[rowTileA][colTileA] = 0.0f;
        }

#pragma unroll
        for (unsigned threadOffset = 0U; threadOffset < BN*BK; threadOffset += NUM_THREADS)
        {
            const unsigned rowTileB = (threadIdx.x + threadOffset) / BN;
            const unsigned colTileB = (threadIdx.x + threadOffset) % BN;
            const unsigned rowB = blockOffset + rowTileB;
            const unsigned colB = bx * BN + colTileB;

            if(rowB < K && colB < N)
                smemB[rowTileB][colTileB] = B[rowB * N + colB];
            else
                smemB[rowTileB][colTileB] = 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (unsigned k = 0U; k < BK; ++k)
            sum += smemA[ty][k] * smemB[k][tx];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// BM % TM == 0U && (BM * BN / TM) % BN == 0U && (BM * BN / TM) % BK == 0U &&
// BN * BK % (BM * BN / TM) == 0U && BM * BK % (BM * BN / TM) == 0U
template <const unsigned BM, const unsigned BN, const unsigned BK, const unsigned TM>
__global__ void matMul_tiled_1D_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned NUM_THREADS = BM * BN / TM;

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

    const unsigned ty = threadIdx.x / BN * TM;
    const unsigned tx = threadIdx.x % BN;

    const unsigned row = by * BM + ty;
    const unsigned col = bx * BN + tx;

    __shared__ float smemA[BK][BM];
    __shared__ float smemB[BK][BN];

    float sums[TM] = {0.0f};

    for (unsigned blockOffset = 0U; blockOffset < K; blockOffset += BK)
    {
#pragma unroll
        for (unsigned threadOffset = 0U; threadOffset < BM * BK; threadOffset += NUM_THREADS)
        {
            const unsigned rowTileA = (threadIdx.x + threadOffset) / BK;
            const unsigned colTileA = (threadIdx.x + threadOffset) % BK;
            const unsigned rowA = by * BM + rowTileA;
            const unsigned colA = blockOffset + colTileA;
            
            if(rowA < M && colA < K)
                smemA[colTileA][rowTileA] = A[rowA * K + colA];
            else
                smemA[colTileA][rowTileA] = 0.0f;
        }

#pragma unroll
        for (unsigned threadOffset = 0U; threadOffset < BN*BK; threadOffset += NUM_THREADS)
        {
            const unsigned rowTileB = (threadIdx.x + threadOffset) / BN;
            const unsigned colTileB = (threadIdx.x + threadOffset) % BN;
            const unsigned rowB = blockOffset + rowTileB;
            const unsigned colB = bx * BN + colTileB;

            if(rowB < K && colB < N)
                smemB[rowTileB][colTileB] = B[rowB * N + colB];
            else
                smemB[rowTileB][colTileB] = 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (unsigned k = 0U; k < BK; ++k)
        {
            float tmp = smemB[k][tx];
            for (unsigned m = 0U; m < TM; ++m)
                sums[m] += smemA[k][ty + m] * tmp;
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned m = 0U; m < TM; ++m)
    {
        if ((row + m) < M && col < N)
            C[(row + m) * N + col] = sums[m];
    }
}

// BM % TM == 0U && BN % TN == 0U && ((BM * BN) / (TM * TN)) % BN == 0U && ((BM * BN) / (TM * TN)) % BK == 0U &&
// BN * BK % ((BM * BN) / (TM * TN)) == 0U && BM * BK % ((BM * BN) / (TM * TN)) == 0U
template <const unsigned BM, const unsigned BN, const unsigned BK, const unsigned TM, const unsigned TN>
__global__ void matMul_tiled_2D_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned NUM_THREADS = BM * BN / (TM * TN);

    const unsigned ty = (threadIdx.x / (BN / TN)) * TM;
    const unsigned tx = (threadIdx.x % (BN / TN)) * TN;

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

    const unsigned row = by * BM + ty;
    const unsigned col = bx * BN + tx;

    __shared__ float smemA[BK][BM];
    __shared__ float smemB[BK][BN];

    float sums[TM][TN] = {0.0f};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (unsigned blockOffset = 0U; blockOffset < K; blockOffset += BK)
    {
#pragma unroll
        for (unsigned threadOffset = 0U; threadOffset < BM * BK; threadOffset += NUM_THREADS)
        {
            const unsigned rowTileA = (threadIdx.x + threadOffset) / BK;
            const unsigned colTileA = (threadIdx.x + threadOffset) % BK;
            const unsigned rowA = by * BM + rowTileA;
            const unsigned colA = blockOffset + colTileA;
            
            if(rowA < M && colA < K)
                smemA[colTileA][rowTileA] = A[rowA * K + colA];
            else
                smemA[colTileA][rowTileA] = 0.0f;
        }

#pragma unroll
        for (unsigned threadOffset = 0U; threadOffset < BN*BK; threadOffset += NUM_THREADS)
        {
            const unsigned rowTileB = (threadIdx.x + threadOffset) / BN;
            const unsigned colTileB = (threadIdx.x + threadOffset) % BN;
            const unsigned rowB = blockOffset + rowTileB;
            const unsigned colB = bx * BN + colTileB;

            if(rowB < K && colB < N)
                smemB[rowTileB][colTileB] = B[rowB * N + colB];
            else
                smemB[rowTileB][colTileB] = 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (unsigned k = 0U; k < BK; ++k)
        {
            for (unsigned m = 0U; m < TM; ++m)
                regM[m] = smemA[k][ty + m];

            for (unsigned n = 0U; n < TN; ++n)
                regN[n] = smemB[k][tx + n];

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
        for (unsigned n = 0U; n < TN; ++n)
        {
            if((row + m) < M && col + n < N)
                C[(row + m) * N + col + n] = sums[m][n];
        }
    }
}

// BM % TM == 0U && BN % TN == 0U && ((BM * BN) / (TM * TN)) % BN == 0U && ((BM * BN) / (TM * TN)) % BK == 0U &&
// BN * BK % ((BM * BN) / (TM * TN)) == 0U && BM * BK % ((BM * BN) / (TM * TN)) == 0U &&
// K % 4 == 0U && BK % 4 == 0U && BN % 4 == 0U && N % 4 == 0U && TM % 4 == 0U && TN % 4 == 0U
template <const unsigned BM, const unsigned BN, const unsigned BK, const unsigned TM, const unsigned TN>
__global__ void matMul_tiled4_2D_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned NUM_THREADS = BM * BN / (TM * TN);

    const unsigned ty = (threadIdx.x / (BN / TN)) * TM;
    const unsigned tx = (threadIdx.x % (BN / TN)) * TN;

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

    const unsigned row = by * BM + ty;
    const unsigned col = bx * BN + tx;

    __shared__ float smemA[BK][BM];
    __shared__ float smemB[BK][BN];

    float sums[TM][TN] = {0.0f};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    constexpr unsigned VEC_SIZE = 4U;
    constexpr unsigned BK4 = BK / VEC_SIZE;
    constexpr unsigned BN4 = BN / VEC_SIZE;

    for (unsigned blockOffset = 0U; blockOffset < K; blockOffset += BK)
    {
#pragma unroll
        for (unsigned threadOffset = 0U; threadOffset < BM * BK4; threadOffset += NUM_THREADS)
        {
            const unsigned rowTileA = (threadIdx.x + threadOffset) / BK4;
            const unsigned colTileA = (threadIdx.x + threadOffset) % BK4 * VEC_SIZE;
            const unsigned rowA = by * BM + rowTileA;
            const unsigned colA = blockOffset + colTileA;

            float4 tmpA = {0.0f, 0.0f, 0.0f, 0.0f};
            if(rowA < M && colA < K)
                tmpA = *reinterpret_cast<const float4*>(&A[rowA * K + colA]);

            for(unsigned i = 0U; i < VEC_SIZE; ++i)
                smemA[colTileA + i][rowTileA] = reinterpret_cast<float*>(&tmpA)[i];
        }

#pragma unroll
        for (unsigned threadOffset = 0U; threadOffset < BN4*BK; threadOffset += NUM_THREADS)
        {
            const unsigned rowTileB = (threadIdx.x + threadOffset) / BN4;
            const unsigned colTileB = (threadIdx.x + threadOffset) % BN4 * VEC_SIZE;
            const unsigned rowB = blockOffset + rowTileB;
            const unsigned colB = bx * BN + colTileB;

            float4 tmpB = {0.0f, 0.0f, 0.0f, 0.0f};
            if(rowB < K && colB < N)
                tmpB = *reinterpret_cast<const float4*>(&B[rowB * N + colB]);
            *reinterpret_cast<float4*>(&smemB[rowTileB][colTileB]) = tmpB;
        }
        __syncthreads();

#pragma unroll
        for (unsigned k = 0U; k < BK; ++k)
        {
            for (unsigned m = 0U; m < TM; m += VEC_SIZE)
                *reinterpret_cast<float4*>(&regM[m]) = *reinterpret_cast<float4*>(&smemA[k][ty + m]);

            for (unsigned n = 0U; n < TN; n += VEC_SIZE)
                *reinterpret_cast<float4*>(&regN[n]) = *reinterpret_cast<float4*>(&smemB[k][tx + n]);

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
        for (unsigned n = 0U; n < TN; n += VEC_SIZE)
        {
            if((row + m) < M && col + n < N)
                *reinterpret_cast<float4*>(&C[(row + m) * N + col + n]) = *reinterpret_cast<float4*>(&sums[m][n]);
        }
    }
}

///// Launch functions /////

template <unsigned blockDimX = 16U, unsigned blockDimY = 16U>
void launch_matMul_naive(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const dim3 blockDim(blockDimX, blockDimY);
    const dim3 gridDim(CEIL_DIV(M, blockDim.x), CEIL_DIV(N, blockDim.y));

    matMul_naive_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

template <unsigned blockDimX = 16U, unsigned blockDimY = 16U>
void launch_matMul_coalescing(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const dim3 blockDim(blockDimX, blockDimY);
    const dim3 gridDim(CEIL_DIV(N, blockDim.x), CEIL_DIV(M, blockDim.y));

    matMul_coalescing_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

template <const unsigned BM = 16U, const unsigned BN = 16U, const unsigned BK = 16U>
void launch_matMul_tiled(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned NUM_THREADS = BM * BN;

    static_assert(BM * BK % NUM_THREADS == 0U);
    static_assert(BK * BN % NUM_THREADS == 0U);

    const dim3 blockDim(NUM_THREADS);
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    matMul_tiled_kernel<BM, BN, BK><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

template <const unsigned BM = 32U, const unsigned BN = 32U, const unsigned BK = 8U, const unsigned TM = 8U>
void launch_matMul_tiled_1D(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned NUM_THREADS = BM * BN / TM;

    static_assert(BM % TM == 0U);
    static_assert(NUM_THREADS % BN == 0U);
    static_assert(NUM_THREADS % BK == 0U);
    static_assert(BM * BK % NUM_THREADS == 0U);
    static_assert(BK * BN % NUM_THREADS == 0U);

    const dim3 blockDim(NUM_THREADS);
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    matMul_tiled_1D_kernel<BM, BN, BK, TM><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

template <const unsigned BM = 128U, const unsigned BN = 128U, const unsigned BK = 8U, const unsigned TM = 8U, const unsigned TN = 8U>
void launch_matMul_tiled_2D(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned NUM_THREADS = BM * BN / (TM * TN);

    static_assert(BM % TM == 0U);
    static_assert(BN % TN == 0U);
    static_assert(NUM_THREADS % BN == 0U);
    static_assert(NUM_THREADS % BK == 0U);
    static_assert(BM * BK % NUM_THREADS == 0U);
    static_assert(BK * BN % NUM_THREADS == 0U);

    const dim3 blockDim(NUM_THREADS);
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    if(N % 4U == 0U && K % 4U == 0U &&
       BK % 4U == 0U && BN % 4U == 0U &&
       TM % 4U == 0U && TN % 4U == 0U &&
       (BM * BK / 4U) % NUM_THREADS == 0U && (BN * BK / 4U) % NUM_THREADS == 0U
      )
        matMul_tiled4_2D_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    else
        matMul_tiled_2D_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

#endif // MATRIX_MUL_KERNELS
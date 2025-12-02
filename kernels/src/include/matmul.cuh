#ifndef MATMUL_KERNELS
#define MATMUL_KERNELS

#include <util.cuh>

template <unsigned BM, unsigned BN, unsigned BK, unsigned NUM_THREADS>
__device__ void loadToShared(const float *A, const float *B, unsigned M, unsigned N, unsigned K,
     float smemA[BM][BK], float smemB[BK][BN], unsigned threadIdx, unsigned blockOffset)
{
    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

#pragma unroll
    for (unsigned threadOffset = 0U; threadOffset < BM * BK; threadOffset += NUM_THREADS)
    {
        const unsigned rowTileA = (threadIdx + threadOffset) / BK;
        const unsigned colTileA = (threadIdx + threadOffset) % BK;
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
        const unsigned rowTileB = (threadIdx + threadOffset) / BN;
        const unsigned colTileB = (threadIdx + threadOffset) % BN;
        const unsigned rowB = blockOffset + rowTileB;
        const unsigned colB = bx * BN + colTileB;

        if(rowB < K && colB < N)
            smemB[rowTileB][colTileB] = B[rowB * N + colB];
        else
            smemB[rowTileB][colTileB] = 0.0f;
    }
}

template <unsigned BM, unsigned BN, unsigned BK, unsigned NUM_THREADS>
__device__ void loadToSharedTrans(const float *A, const float *B, unsigned M, unsigned N, unsigned K,
     float smemA[BK][BM], float smemB[BK][BN], unsigned threadIdx, unsigned blockOffset)
{
    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

#pragma unroll
    for (unsigned threadOffset = 0U; threadOffset < BM * BK; threadOffset += NUM_THREADS)
    {
        const unsigned rowTileA = (threadIdx + threadOffset) / BK;
        const unsigned colTileA = (threadIdx + threadOffset) % BK;
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
        const unsigned rowTileB = (threadIdx + threadOffset) / BN;
        const unsigned colTileB = (threadIdx + threadOffset) % BN;
        const unsigned rowB = blockOffset + rowTileB;
        const unsigned colB = bx * BN + colTileB;

        if(rowB < K && colB < N)
            smemB[rowTileB][colTileB] = B[rowB * N + colB];
        else
            smemB[rowTileB][colTileB] = 0.0f;
    }
}

template <unsigned BM, unsigned BN, unsigned BK, unsigned NUM_THREADS>
__device__ void loadToSharedTransVect(const float *A, const float *B, unsigned M, unsigned N, unsigned K,
     float smemA[BK][BM], float smemB[BK][BN], unsigned threadIdx, unsigned blockOffset)
{
    constexpr unsigned VEC_SIZE = 4U;
    constexpr unsigned BK4 = BK / VEC_SIZE;
    constexpr unsigned BN4 = BN / VEC_SIZE;

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

#pragma unroll
    for (unsigned threadOffset = 0U; threadOffset < BM * BK4; threadOffset += NUM_THREADS)
    {
        const unsigned rowTileA = (threadIdx + threadOffset) / BK4;
        const unsigned colTileA = (threadIdx + threadOffset) % BK4 * VEC_SIZE;
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
        const unsigned rowTileB = (threadIdx + threadOffset) / BN4;
        const unsigned colTileB = (threadIdx + threadOffset) % BN4 * VEC_SIZE;
        const unsigned rowB = blockOffset + rowTileB;
        const unsigned colB = bx * BN + colTileB;

        float4 tmpB = {0.0f, 0.0f, 0.0f, 0.0f};
        if(rowB < K && colB < N)
            tmpB = *reinterpret_cast<const float4*>(&B[rowB * N + colB]);
        *reinterpret_cast<float4*>(&smemB[rowTileB][colTileB]) = tmpB;
    }
}

template <unsigned BM, unsigned BN, unsigned BK>
__device__ void processBlockTiles(unsigned ty, unsigned tx, const float smemA[BM][BK], const float smemB[BK][BN], float& sum)
{
#pragma unroll
    for (unsigned k = 0U; k < BK; ++k)
        sum += smemA[ty][k] * smemB[k][tx];
}

template <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN>
__device__ void processThreadTransTiles(unsigned ty, unsigned tx, const float smemA[BK][BM], const float smemB[BK][BN],
    float regM[TM], float regN[TN], float sums[TM][TN])
{
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
}

template <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN>
__device__ void processThreadTransTilesVect(unsigned ty, unsigned tx, const float smemA[BK][BM], const float smemB[BK][BN],
    float regM[TM], float regN[TN], float sums[TM][TN])
{
    constexpr unsigned VEC_SIZE = 4U;

#pragma unroll
    for (unsigned k = 0U; k < BK; ++k)
    {
        for (unsigned m = 0U; m < TM; m += VEC_SIZE)
            *reinterpret_cast<float4*>(&regM[m]) = *reinterpret_cast<const float4*>(&smemA[k][ty + m]);

        for (unsigned n = 0U; n < TN; n += VEC_SIZE)
            *reinterpret_cast<float4*>(&regN[n]) = *reinterpret_cast<const float4*>(&smemB[k][tx + n]);

        for (unsigned m = 0U; m < TM; ++m)
        {
            for (unsigned n = 0; n < TN; ++n)
                sums[m][n] += regM[m] * regN[n];
        }
    }
}

__global__ void matmul_naive_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K);
__global__ void matmul_coalescing_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K);

template <unsigned BM, unsigned BN, unsigned BK>
__global__ void matmul_BTiles_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned NUM_THREADS = BM * BN;

    static_assert(BM * BK % NUM_THREADS == 0U);
    static_assert(BK * BN % NUM_THREADS == 0U);

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
        loadToShared<BM, BN, BK, NUM_THREADS>(A, B, M, N, K, smemA, smemB, threadIdx.x, blockOffset);
        __syncthreads();

        processBlockTiles<BM, BN, BK>(ty, tx, smemA, smemB, sum);
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

template <unsigned BM, unsigned BN, unsigned BK>
__global__ void matmul_BTiles_DBuf_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned NUM_THREADS = BM * BN;
    constexpr unsigned NUM_THREADS2 = NUM_THREADS / 2U;

    static_assert(BM * BK % NUM_THREADS == 0U);
    static_assert(BK * BN % NUM_THREADS == 0U);

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;
    
    const unsigned ty = threadIdx.x / BN;
    const unsigned tx = threadIdx.x % BN;

    const unsigned row = by * BM + ty;
    const unsigned col = bx * BN + tx;

    __shared__ float smemA[2U][BM][BK];
    __shared__ float smemB[2U][BK][BN];

    float sum = 0.0f;

    unsigned doubleBufferIdx = static_cast<unsigned>(threadIdx.x >= NUM_THREADS2);

    if(doubleBufferIdx == 0U)
        loadToShared<BM, BN, BK, NUM_THREADS2>(A, B, M, N, K, smemA[0U], smemB[0U], threadIdx.x, 0U);
    __syncthreads();

    for (unsigned blockOffset = 0U; blockOffset < K; blockOffset += 2U * BK)
    {
        if(doubleBufferIdx == 0U)
        {
            processBlockTiles<BM, BN, BK>(ty, tx, smemA[0U], smemB[0U], sum);
            __syncthreads();

            if(blockOffset + BK < K)
                processBlockTiles<BM, BN, BK>(ty, tx, smemA[1U], smemB[1U], sum);
            __syncthreads();

            if(blockOffset + 2U * BK < K)
                loadToShared<BM, BN, BK, NUM_THREADS2>(A, B, M, N, K, smemA[0U], smemB[0U], threadIdx.x, blockOffset + 2U * BK);
            __syncthreads();
        }
        else
        {
            if(blockOffset + BK < K)
                loadToShared<BM, BN, BK, NUM_THREADS2>(A, B, M, N, K, smemA[1U], smemB[1U], threadIdx.x - NUM_THREADS2, blockOffset + BK);
            __syncthreads();

            processBlockTiles<BM, BN, BK>(ty, tx, smemA[0U], smemB[0U], sum);
            __syncthreads();

            if(blockOffset + BK < K)
                processBlockTiles<BM, BN, BK>(ty, tx, smemA[1U], smemB[1U], sum);
            __syncthreads();
        }
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

template <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN>
__global__ void matmul_TTiles_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned NUM_THREADS = BM * BN / (TM * TN);

    static_assert(BM % TM == 0U);
    static_assert(BN % TN == 0U);
    // static_assert(NUM_THREADS % BN == 0U);
    // static_assert(NUM_THREADS % BK == 0U);
    static_assert(BM * BK % NUM_THREADS == 0U);
    static_assert(BK * BN % NUM_THREADS == 0U);

    const unsigned ty = (threadIdx.x / (BN / TN)) * TM;
    const unsigned tx = (threadIdx.x % (BN / TN)) * TN;

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

    const unsigned row = by * BM + ty;
    const unsigned col = bx * BN + tx;

    __shared__ float smemA[BK][BM];  // transposed
    __shared__ float smemB[BK][BN];

    float sums[TM][TN] = {0.0f};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (unsigned blockOffset = 0U; blockOffset < K; blockOffset += BK)
    {
        loadToSharedTrans<BM, BN, BK, NUM_THREADS>(A, B, M, N, K, smemA, smemB, threadIdx.x, blockOffset);
        __syncthreads();

        processThreadTransTiles<BM, BN, BK, TM, TN>(ty, tx, smemA, smemB, regM, regN, sums);
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

template <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN>
__global__ void matmul_TTiles_DBuf_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned NUM_THREADS = BM * BN / (TM * TN);
    constexpr unsigned NUM_THREADS2 = NUM_THREADS / 2U;

    static_assert(BM % TM == 0U);
    static_assert(BN % TN == 0U);
    // static_assert(NUM_THREADS % BN == 0U);
    // static_assert(NUM_THREADS % BK == 0U);
    static_assert(BM * BK % NUM_THREADS == 0U);
    static_assert(BK * BN % NUM_THREADS == 0U);

    const unsigned ty = (threadIdx.x / (BN / TN)) * TM;
    const unsigned tx = (threadIdx.x % (BN / TN)) * TN;

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

    const unsigned row = by * BM + ty;
    const unsigned col = bx * BN + tx;

    __shared__ float smemA[2U][BK][BM];  // transposed
    __shared__ float smemB[2U][BK][BN];

    float sums[TM][TN] = {0.0f};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    unsigned doubleBufferIdx = static_cast<unsigned>(threadIdx.x >= NUM_THREADS2);

    if(doubleBufferIdx == 0U)
        loadToSharedTrans<BM, BN, BK, NUM_THREADS2>(A, B, M, N, K, smemA[0U], smemB[0U], threadIdx.x, 0U);
    __syncthreads();

    for (unsigned blockOffset = 0U; blockOffset < K; blockOffset += 2U * BK)
    {
        if(doubleBufferIdx == 0U)
        {
            processThreadTransTiles<BM, BN, BK, TM, TN>(ty, tx, smemA[0U], smemB[0U], regM, regN, sums);
            __syncthreads();

            if(blockOffset + BK < K)
                processThreadTransTiles<BM, BN, BK, TM, TN>(ty, tx, smemA[1U], smemB[1U], regM, regN, sums);
            __syncthreads();

            if(blockOffset + 2U * BK < K)
                loadToSharedTrans<BM, BN, BK, NUM_THREADS2>(A, B, M, N, K, smemA[0U], smemB[0U], threadIdx.x, blockOffset + 2U * BK);
            __syncthreads();
        }
        else
        {
            if(blockOffset + BK < K)
                loadToSharedTrans<BM, BN, BK, NUM_THREADS2>(A, B, M, N, K, smemA[1U], smemB[1U], threadIdx.x - NUM_THREADS2, blockOffset + BK);
            __syncthreads();

            processThreadTransTiles<BM, BN, BK, TM, TN>(ty, tx, smemA[0U], smemB[0U], regM, regN, sums);
            __syncthreads();

            if(blockOffset + BK < K)
                processThreadTransTiles<BM, BN, BK, TM, TN>(ty, tx, smemA[1U], smemB[1U], regM, regN, sums);
            __syncthreads();
        }
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

// K % 4 == 0U && N % 4 == 0U
template <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN>
__global__ void matmul_TTiles_vec_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned NUM_THREADS = BM * BN / (TM * TN);
    constexpr unsigned VEC_SIZE = 4U;

    static_assert(BM % TM == 0U);
    static_assert(BN % TN == 0U);
    // static_assert(NUM_THREADS % BN == 0U);
    // static_assert(NUM_THREADS % BK == 0U);
    static_assert(BM * BK % NUM_THREADS == 0U);
    static_assert(BK * BN % NUM_THREADS == 0U);
    static_assert(BK % VEC_SIZE == 0U);
    static_assert(BN % VEC_SIZE == 0U);
    static_assert(TM % VEC_SIZE == 0U);
    static_assert(TN % VEC_SIZE == 0U);
    static_assert((BM * BK / VEC_SIZE) % NUM_THREADS == 0U);
    static_assert((BN * BK / VEC_SIZE) % NUM_THREADS == 0U);

    const unsigned ty = (threadIdx.x / (BN / TN)) * TM;
    const unsigned tx = (threadIdx.x % (BN / TN)) * TN;

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

    const unsigned row = by * BM + ty;
    const unsigned col = bx * BN + tx;

    __shared__ float smemA[BK][BM];  // transposed
    __shared__ float smemB[BK][BN];

    float sums[TM][TN] = {0.0f};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (unsigned blockOffset = 0U; blockOffset < K; blockOffset += BK)
    {
        loadToSharedTransVect<BM, BN, BK, NUM_THREADS>(A, B, M, N, K, smemA, smemB, threadIdx.x, blockOffset);
        __syncthreads();

        processThreadTransTilesVect<BM, BN, BK, TM, TN>(ty, tx, smemA, smemB, regM, regN, sums);
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

template <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN>
__global__ void matmul_TTiles_DBuf_vec_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    constexpr unsigned NUM_THREADS = BM * BN / (TM * TN);
    constexpr unsigned NUM_THREADS2 = NUM_THREADS / 2U;
    constexpr unsigned VEC_SIZE = 4U;

    static_assert(BM % TM == 0U);
    static_assert(BN % TN == 0U);
    // static_assert(NUM_THREADS % BN == 0U);
    // static_assert(NUM_THREADS % BK == 0U);
    static_assert(BM * BK % NUM_THREADS == 0U);
    static_assert(BK * BN % NUM_THREADS == 0U);
    static_assert(BK % VEC_SIZE == 0U);
    static_assert(BN % VEC_SIZE == 0U);
    static_assert(TM % VEC_SIZE == 0U);
    static_assert(TN % VEC_SIZE == 0U);
    static_assert((BM * BK / VEC_SIZE) % NUM_THREADS == 0U);
    static_assert((BN * BK / VEC_SIZE) % NUM_THREADS == 0U);

    const unsigned ty = (threadIdx.x / (BN / TN)) * TM;
    const unsigned tx = (threadIdx.x % (BN / TN)) * TN;

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

    const unsigned row = by * BM + ty;
    const unsigned col = bx * BN + tx;

    __shared__ float smemA[2U][BK][BM];  // transposed
    __shared__ float smemB[2U][BK][BN];

    float sums[TM][TN] = {0.0f};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    unsigned doubleBufferIdx = static_cast<unsigned>(threadIdx.x >= NUM_THREADS2);

    if(doubleBufferIdx == 0U)
        loadToSharedTransVect<BM, BN, BK, NUM_THREADS2>(A, B, M, N, K, smemA[0U], smemB[0U], threadIdx.x, 0U);
    __syncthreads();

    for (unsigned blockOffset = 0U; blockOffset < K; blockOffset += 2U * BK)
    {
        if(doubleBufferIdx == 0U)
        {
            processThreadTransTilesVect<BM, BN, BK, TM, TN>(ty, tx, smemA[0U], smemB[0U], regM, regN, sums);
            __syncthreads();

            if(blockOffset + BK < K)
                processThreadTransTilesVect<BM, BN, BK, TM, TN>(ty, tx, smemA[1U], smemB[1U], regM, regN, sums);
            __syncthreads();

            if(blockOffset + 2U * BK < K)
                loadToSharedTransVect<BM, BN, BK, NUM_THREADS2>(A, B, M, N, K, smemA[0U], smemB[0U], threadIdx.x, blockOffset + 2U * BK);
            __syncthreads();
        }
        else
        {
            if(blockOffset + BK < K)
                loadToSharedTransVect<BM, BN, BK, NUM_THREADS2>(A, B, M, N, K, smemA[1U], smemB[1U], threadIdx.x - NUM_THREADS2, blockOffset + BK);
            __syncthreads();

            processThreadTransTilesVect<BM, BN, BK, TM, TN>(ty, tx, smemA[0U], smemB[0U], regM, regN, sums);
            __syncthreads();

            if(blockOffset + BK < K)
                processThreadTransTilesVect<BM, BN, BK, TM, TN>(ty, tx, smemA[1U], smemB[1U], regM, regN, sums);
            __syncthreads();
        }
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
void launch_matmul_naive(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const dim3 blockDim(blockDimX, blockDimY);
    const dim3 gridDim(CEIL_DIV(M, blockDim.x), CEIL_DIV(N, blockDim.y));

    matmul_naive_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

template <unsigned blockDimX = 16U, unsigned blockDimY = 16U>
void launch_matmul_coalescing(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const dim3 blockDim(blockDimX, blockDimY);
    const dim3 gridDim(CEIL_DIV(N, blockDim.x), CEIL_DIV(M, blockDim.y));

    matmul_coalescing_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

template <unsigned BM = 16U, unsigned BN = 16U, unsigned BK = 16U>
void launch_matmul_BTiles(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const dim3 blockDim(BM * BN);
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    matmul_BTiles_kernel<BM, BN, BK><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

template <unsigned BM = 16U, unsigned BN = 16U, unsigned BK = 16U>
void launch_matmul_BTiles_DBuf(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const dim3 blockDim(BM * BN);
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    matmul_BTiles_DBuf_kernel<BM, BN, BK><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

template <unsigned BM = 128U, unsigned BN = 128U, unsigned BK = 8U, unsigned TM = 8U, unsigned TN = 8U>
void launch_matmul_TTiles(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const dim3 blockDim(BM * BN / (TM * TN));
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    matmul_TTiles_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

template <unsigned BM = 128U, unsigned BN = 128U, unsigned BK = 8U, unsigned TM = 8U, unsigned TN = 8U>
void launch_matmul_TTiles_DBuf(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const dim3 blockDim(BM * BN / (TM * TN));
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    matmul_TTiles_DBuf_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

template <unsigned BM = 128U, unsigned BN = 128U, unsigned BK = 8U, unsigned TM = 8U, unsigned TN = 8U>
void launch_matmul_TTiles_vec(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const dim3 blockDim(BM * BN / (TM * TN));
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    if(N % 4U == 0U && K % 4U == 0U)
        matmul_TTiles_vec_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    else
        matmul_TTiles_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

template <unsigned BM = 128U, unsigned BN = 128U, unsigned BK = 8U, unsigned TM = 8U, unsigned TN = 8U>
void launch_matmul_TTiles_DBuf_vec(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
    const dim3 blockDim(BM * BN / (TM * TN));
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    if(N % 4U == 0U && K % 4U == 0U)
        matmul_TTiles_DBuf_vec_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    else
        matmul_TTiles_DBuf_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

#endif // MATMUL_KERNELS
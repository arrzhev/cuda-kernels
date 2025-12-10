#ifndef MATMUL_KERNELS
#define MATMUL_KERNELS

#include <util.cuh>

template <typename T>
concept MatType = IsAnyOf<T, int, double, float>;

template <unsigned BM, unsigned BN, unsigned BK, unsigned NUM_THREADS,
          bool LOAD_AT, bool LOAD_BT, bool STORE_AT, bool VEC, MatType T>
__device__ void loadToShared(const T *A, const T *B, unsigned M, unsigned N, unsigned K,
    T* smemA, T* smemB, unsigned threadIdx, unsigned blockOffset)
{
    using VecType = typename std::conditional_t<VEC, int4, T>;

    auto loadMatToShared = [threadIdx] <unsigned B_ROWS, unsigned B_COLS, bool LOAD_T, bool STORE_T> (
        const T *mat, unsigned rows, unsigned cols, T* smem, unsigned rowOffset, unsigned colOffset)
    {
        constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);
        constexpr unsigned B_ROWS_4 = B_ROWS / VEC_SIZE;
        constexpr unsigned B_COLS_4 = B_COLS / VEC_SIZE;

#pragma unroll
        for (unsigned threadOffset = 0U; threadOffset < B_ROWS * B_COLS / VEC_SIZE; threadOffset += NUM_THREADS)
        {
            unsigned rowTile;
            unsigned colTile;

            if constexpr(LOAD_T)
            {
                rowTile = (threadIdx + threadOffset) % B_ROWS_4 * VEC_SIZE;
                colTile = (threadIdx + threadOffset) / B_ROWS_4;
            }
            else
            {
                rowTile = (threadIdx + threadOffset) / B_COLS_4;
                colTile = (threadIdx + threadOffset) % B_COLS_4 * VEC_SIZE;
            }

            const unsigned row = rowOffset + rowTile;
            const unsigned col = colOffset + colTile;

            VecType tmpVal;
            if constexpr(VEC)
                tmpVal = {0, 0, 0, 0};
            else
                tmpVal = static_cast<T>(0);

            if constexpr(LOAD_T)
            {
                if(row < rows && col < cols)
                    tmpVal = *reinterpret_cast<const VecType*>(&mat[col * rows + row]);

                if constexpr(STORE_T)
                {
                    *reinterpret_cast<VecType*>(&smem[colTile * B_ROWS + rowTile]) = tmpVal;
                }
                else
                {
                    for(unsigned i = 0U; i < VEC_SIZE; ++i)
                        smem[(rowTile + i) * B_COLS + colTile] = reinterpret_cast<T*>(&tmpVal)[i];
                }
            }
            else
            {
                if(row < rows && col < cols)
                    tmpVal =  *reinterpret_cast<const VecType*>(&mat[row * cols + col]);

                if constexpr(STORE_T)
                {
                    for(unsigned i = 0U; i < VEC_SIZE; ++i)
                        smem[(colTile + i) * B_ROWS + rowTile] = reinterpret_cast<T*>(&tmpVal)[i];
                }
                else
                    *reinterpret_cast<VecType*>(&smem[rowTile * B_COLS + colTile]) = tmpVal;  
            }
        }
    };

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

    loadMatToShared.template operator()<BM, BK, LOAD_AT, STORE_AT>(A, M, K, smemA, by * BM, blockOffset);

    loadMatToShared.template operator()<BK, BN, LOAD_BT, false>(B, K, N, smemB, blockOffset, bx * BN);
}

template <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN, bool STORE_AT, bool VEC, MatType T>
__device__ void processTiles(unsigned ty, unsigned tx, const T* smemA, const T* smemB, T regM[TM], T regN[TN], T sums[TM][TN])
{
    using VecType = typename std::conditional_t<VEC, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);

#pragma unroll
    for (unsigned k = 0U; k < BK; ++k)
    {
        for (unsigned m = 0U; m < TM; m += VEC_SIZE)
        {
            if constexpr(STORE_AT)
                *reinterpret_cast<VecType*>(&regM[m]) = *reinterpret_cast<const VecType*>(&smemA[k * BM + ty + m]);
            else
            {
                for(unsigned i = 0U; i < VEC_SIZE; ++i)
                    regM[m + i] = smemA[(ty + m + i) * BK + k];
            }
        }

        for (unsigned n = 0U; n < TN; n += VEC_SIZE)
            *reinterpret_cast<VecType*>(&regN[n]) = *reinterpret_cast<const VecType*>(&smemB[k * BN + tx + n]);

        for (unsigned m = 0U; m < TM; ++m)
        {
            for (unsigned n = 0; n < TN; ++n)
                sums[m][n] += regM[m] * regN[n];
        }
    }
}

template <unsigned SK, unsigned TM, unsigned TN, bool VEC, MatType T>
__device__ void storeResult(T *mat, unsigned row, unsigned col, unsigned rows, unsigned cols, const T sums[TM][TN])
{
    using VecType = typename std::conditional_t<VEC && SK == 1U, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);

    for (unsigned m = 0U; m < TM; ++m)
    {
        for (unsigned n = 0U; n < TN; n += VEC_SIZE)
        {
            if((row + m) < rows && col + n < cols)
            {
                if constexpr (SK == 1U)
                    *reinterpret_cast<VecType*>(&mat[(row + m) * cols + col + n]) = *reinterpret_cast<const VecType*>(&sums[m][n]);
                else
                    atomicAdd(&mat[(row + m) * cols + col + n], sums[m][n]);
            }
        }
    }
}

template <unsigned SK, bool COAL, bool AT, bool BT, MatType T>
__global__ void matmul_naive_kernel(const T *A, const T *B, T *C, unsigned M, unsigned N, unsigned K)
{
    unsigned row;
    unsigned col;

    if constexpr(COAL)
    {
        col = blockIdx.x * blockDim.x + threadIdx.x;
        row = blockIdx.y * blockDim.y + threadIdx.y;
    }
    else
    {
        row = blockIdx.x * blockDim.x + threadIdx.x;
        col = blockIdx.y * blockDim.y + threadIdx.y;
    }

    unsigned splitStart;
    unsigned splitEnd;
    bool isInRange;

    if constexpr (SK == 1U)
    {
        splitStart = 0U;
        splitEnd = K;
        isInRange = row < M && col < N;
    }
    else
    {
        const unsigned bz = blockIdx.z;
        const unsigned splitSize = CEIL_DIV(K, SK);
        splitStart = splitSize * bz;
        splitEnd = min(splitStart + splitSize, K);
        isInRange = row < M && col < N && splitStart < K;
    }

    if (isInRange)
    {
        float sum = 0.0f;
        for (unsigned i = splitStart; i < splitEnd; ++i)
        {
            T valA;
            T valB;
            if constexpr(AT)
                valA = A[row + i * M];
            else
                valA = A[row * K + i];

            if constexpr(BT)
                valB = B[i + col * K];
            else
                valB = B[i * N + col];

            sum += valA * valB;
        }

        if constexpr(SK == 1U)
            C[row * N + col] = sum;
        else
            atomicAdd(&C[row * N + col], sum);
    }
}

template <unsigned SK, unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN, bool AT, bool BT, bool VEC, MatType T>
__global__ void matmul_tiles_kernel(const T *A, const T *B, T *C, unsigned M, unsigned N, unsigned K)
{
    using VecType = typename std::conditional_t<VEC, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);
    constexpr bool STORE_AT = TM > 1U;
    constexpr unsigned NUM_THREADS = BM * BN / (TM * TN);

    static_assert(BM % TM == 0U);
    static_assert(BN % TN == 0U);
    static_assert(BM * BK % NUM_THREADS == 0U);
    static_assert(BK * BN % NUM_THREADS == 0U);
    static_assert(BM % VEC_SIZE == 0U);
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

    __shared__ T smemA[BK * BM];  // STORE_AT - transposed or not; size always BK * BM
    __shared__ T smemB[BK * BN];

    T sums[TM][TN] = {0.0f};
    T regM[TM] = {0.0};
    T regN[TN] = {0.0};

    unsigned splitStart;
    unsigned splitEnd;
    if constexpr (SK == 1U)
    {
        splitStart = 0U;
        splitEnd = K;
    }
    else
    {
        unsigned bz = blockIdx.z;
        unsigned splitSize = CEIL_DIV(CEIL_DIV(K, BK), SK);
        splitStart = splitSize * bz * BK;
        splitEnd = min(splitStart + splitSize * BK, K);
    }

    for (unsigned blockOffset = splitStart; blockOffset < splitEnd; blockOffset += BK)
    {
        loadToShared<BM, BN, BK, NUM_THREADS, AT, BT, STORE_AT, VEC>(A, B, M, N, K, smemA, smemB, threadIdx.x, blockOffset);
        __syncthreads();

        processTiles<BM, BN, BK, TM, TN, STORE_AT, VEC>(ty, tx, smemA, smemB, regM, regN, sums);
        __syncthreads();
    }

    storeResult<SK, TM, TN, VEC>(C, row, col, M, N, sums);
}

template <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN, bool AT, bool BT, bool VEC, MatType T>
__global__ void matmul_tiles_SDBuf_kernel(const T *A, const T *B, T *C, unsigned M, unsigned N, unsigned K)
{
    using VecType = typename std::conditional_t<VEC, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);
    constexpr bool STORE_AT = TM > 1U;
    constexpr unsigned NUM_THREADS = BM * BN / (TM * TN);

    static_assert(BM % TM == 0U);
    static_assert(BN % TN == 0U);
    static_assert(BM * BK % NUM_THREADS == 0U);
    static_assert(BK * BN % NUM_THREADS == 0U);
    static_assert(BM % VEC_SIZE == 0U);
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

    __shared__ T smemA[2U][BK * BM];
    __shared__ T smemB[2U][BK * BN];

    T sums[TM][TN] = {0.0f};
    T regM[TM] = {0.0};
    T regN[TN] = {0.0};

    unsigned currentIdx = 0U;
    unsigned nextIdx = 1U;

    loadToShared<BM, BN, BK, NUM_THREADS, AT, BT, STORE_AT, VEC>(A, B, M, N, K, smemA[currentIdx], smemB[currentIdx], threadIdx.x, 0U);
    __syncthreads();

    for (unsigned blockOffset = BK; blockOffset < K; blockOffset += BK)
    {
        loadToShared<BM, BN, BK, NUM_THREADS, AT, BT, STORE_AT, VEC>(A, B, M, N, K, smemA[nextIdx], smemB[nextIdx], threadIdx.x, blockOffset);

        processTiles<BM, BN, BK, TM, TN, STORE_AT, VEC>(ty, tx, smemA[currentIdx], smemB[currentIdx], regM, regN, sums);
        __syncthreads();

        std::swap(currentIdx, nextIdx);
    }

    processTiles<BM, BN, BK, TM, TN, STORE_AT, VEC>(ty, tx, smemA[currentIdx], smemB[currentIdx], regM, regN, sums);

    storeResult<1U, TM, TN, VEC>(C, row, col, M, N, sums);
}

template <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN, bool AT, bool BT, bool VEC, MatType T>
__global__ void matmul_tiles_DBuf_kernel(const T *A, const T *B, T *C, unsigned M, unsigned N, unsigned K)
{
    using VecType = typename std::conditional_t<VEC, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);
    constexpr bool STORE_AT = TM > 1U;
    constexpr unsigned NUM_THREADS = BM * BN / (TM * TN);
    constexpr unsigned NUM_THREADS2 = NUM_THREADS / 2U;

    static_assert(BM % TM == 0U);
    static_assert(BN % TN == 0U);
    static_assert(BM * BK % NUM_THREADS == 0U);
    static_assert(BK * BN % NUM_THREADS == 0U);
    static_assert(BM % VEC_SIZE == 0U);
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

    __shared__ T smemA[2U][BK * BM];  // transposed
    __shared__ T smemB[2U][BK * BN];

    T sums[TM][TN] = {0.0f};
    T regM[TM] = {0.0};
    T regN[TN] = {0.0};

    unsigned doubleBufferIdx = static_cast<unsigned>(threadIdx.x >= NUM_THREADS2);

    if(doubleBufferIdx == 0U)
        loadToShared<BM, BN, BK, NUM_THREADS2, AT, BT, STORE_AT, VEC>(A, B, M, N, K, smemA[0U], smemB[0U], threadIdx.x, 0U);
    __syncthreads();

    for (unsigned blockOffset = 0U; blockOffset < K; blockOffset += 2U * BK)
    {
        if(doubleBufferIdx == 0U)
        {
            processTiles<BM, BN, BK, TM, TN, STORE_AT, VEC>(ty, tx, smemA[0U], smemB[0U], regM, regN, sums);
            __syncthreads();

            if(blockOffset + BK < K)
                processTiles<BM, BN, BK, TM, TN, STORE_AT, VEC>(ty, tx, smemA[1U], smemB[1U], regM, regN, sums);
            __syncthreads();

            if(blockOffset + 2U * BK < K)
                loadToShared<BM, BN, BK, NUM_THREADS2, AT, BT, STORE_AT, VEC>(A, B, M, N, K, smemA[0U], smemB[0U], threadIdx.x, blockOffset + 2U * BK);
            __syncthreads();
        }
        else
        {
            if(blockOffset + BK < K)
                loadToShared<BM, BN, BK, NUM_THREADS2, AT, BT, STORE_AT, VEC>(A, B, M, N, K, smemA[1U], smemB[1U], threadIdx.x - NUM_THREADS2, blockOffset + BK);
            __syncthreads();

            processTiles<BM, BN, BK, TM, TN, STORE_AT, VEC>(ty, tx, smemA[0U], smemB[0U], regM, regN, sums);
            __syncthreads();

            if(blockOffset + BK < K)
                processTiles<BM, BN, BK, TM, TN, STORE_AT, VEC>(ty, tx, smemA[1U], smemB[1U], regM, regN, sums);
            __syncthreads();
        }
    }

    storeResult<1U, TM, TN, VEC>(C, row, col, M, N, sums);
}

///// Launch functions /////

template <unsigned blockDimX = 16U, unsigned blockDimY = 16U, bool COAL = false, unsigned SK = 1U>
void launch_matmul_naive(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K, bool AT, bool BT)
{
    const dim3 blockDim(blockDimX, blockDimY);
    const dim3 gridDim(CEIL_DIV(COAL ? N : M, blockDim.x), CEIL_DIV(COAL ? M : N, blockDim.y), SK);

    auto launchKernel = [&] <bool T1, bool T2> () {
            matmul_naive_kernel<SK, COAL, T1, T2><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    };

    if(AT && BT)
        launchKernel.template operator()<true, true>();
    else if(AT)
        launchKernel.template operator()<true, false>();
    else if(BT)
        launchKernel.template operator()<false, true>();
    else
        launchKernel.template operator()<false, false>();
}

template <unsigned BM = 128U, unsigned BN = 128U, unsigned BK = 8U, unsigned TM = 8U, unsigned TN = 8U,
          bool DBUF = true, bool VEC = true, unsigned SK = 1U>
void launch_matmul_tiles(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K, bool AT, bool BT)
{
    static_assert(!(DBUF && (SK > 1U))); // Split K not implemented for double buffer

    using VecType = typename std::conditional_t<VEC, int4, float>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(float);

    const dim3 blockDim(BM * BN / (TM * TN));
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), SK);

    auto launchKernel = [&] <bool T1, bool T2, bool V> () {
        if constexpr (DBUF)
            matmul_tiles_DBuf_kernel<BM, BN, BK, TM, TN, T1, T2, V><<<gridDim, blockDim>>>(A, B, C, M, N, K);
        else
            matmul_tiles_kernel<SK, BM, BN, BK, TM, TN, T1, T2, V><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    };

    if(N % VEC_SIZE == 0U && K % VEC_SIZE == 0U)
    {
        if(AT && BT)
            launchKernel.template operator()<true, true, VEC>();
        else if(AT)
            launchKernel.template operator()<true, false, VEC>();
        else if(BT)
            launchKernel.template operator()<false, true, VEC>();
        else
            launchKernel.template operator()<false, false, VEC>();
    }
    else
    {
        if(AT && BT)
            launchKernel.template operator()<true, true, false>();
        else if(AT)
            launchKernel.template operator()<true, false, false>();
        else if(BT)
            launchKernel.template operator()<false, true, false>();
        else
            launchKernel.template operator()<false, false, false>();
    }
}

#endif // MATMUL_KERNELS
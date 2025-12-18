#ifndef MATMUL_KERNELS
#define MATMUL_KERNELS

#include <variant>
#include <util.cuh>
#include <cuda_fp16.h>
#include <mma.h>

// Type of accumulator to store dot product of each element
template <typename T>
concept AccumType = IsAnyOf<T, int, double, float>;

///// Helper functions /////

// Load blocks (BMxBK and BKxBN) of A, B matrices from global memory to shared memory
// AT, BT - specify layout of A, B matrices; true - column major, false - row major
// STOR_AT - indicate if shared memory block for matrix A is transposed
// VEC - indicate if vectorized memory access allowed
template <unsigned BM, unsigned BN, unsigned BK, unsigned NUM_THREADS,
          bool AT, bool BT, bool STORE_AT, bool VEC, InputType T>
__device__ void loadToShared(const T *A, const T *B, unsigned M, unsigned N, unsigned K,
    T* smemA, T* smemB, unsigned threadIdx, unsigned blockOffset)
{
    using VecType = typename std::conditional_t<VEC, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);

    // Load block (B_ROWS x B_COLS) of single matrix A or B from global memory to shared memory
    auto loadMatToShared = [&] <unsigned B_ROWS, unsigned B_COLS, bool LOAD_T, bool STORE_T> (
        const T *mat, unsigned rows, unsigned cols, T* smem, unsigned rowOffset, unsigned colOffset)
    {
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
#pragma unroll
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
#pragma unroll
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

    // Load block (BM X BK) of matrix A[by*BM][blockOffset] from global memory to shared
    loadMatToShared.template operator()<BM, BK, AT, STORE_AT>(A, M, K, smemA, by * BM, blockOffset);

    // Load block (BK X BN) of matrix N[blockOffset][bx*BN] from global memory to shared
    loadMatToShared.template operator()<BK, BN, BT, false>(B, K, N, smemB, blockOffset, bx * BN);
}

// Compute dot product within blocks (BMxBK and BKxBN) in shared memory and store sums in register
// Each thread compute TMxTN elements
template <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN,
          bool STORE_AT, bool CT, bool VEC, InputType T, AccumType SumType>
__device__ void processTiles(unsigned ty, unsigned tx, const T* smemA, const T* smemB, T regM[TM], T regN[TN], SumType* sums)
{
    using VecType = typename std::conditional_t<VEC, int4, T>;
    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);

#pragma unroll
    for (unsigned k = 0U; k < BK; ++k)
    {
        // Save shared memory block for matrix A to registers
#pragma unroll
        for (unsigned m = 0U; m < TM; m += VEC_SIZE)
        {
            // Transposed shared memory block for matrix A allows vectorized memory access
            if constexpr(STORE_AT)
                *reinterpret_cast<VecType*>(&regM[m]) = *reinterpret_cast<const VecType*>(&smemA[k * BM + ty + m]);
            else
            {
                for(unsigned i = 0U; i < VEC_SIZE; ++i)
                    regM[m + i] = smemA[(ty + m + i) * BK + k];
            }
        }

        // Save shared memory block for matrix B to registers
#pragma unroll
        for (unsigned n = 0U; n < TN; n += VEC_SIZE)
            *reinterpret_cast<VecType*>(&regN[n]) = *reinterpret_cast<const VecType*>(&smemB[k * BN + tx + n]);

        // Write from individual registers to sums
        // Transpose sums when matrix C is column major for better memory access in store function
        if constexpr (CT)
        {
#pragma unroll
            for (unsigned n = 0; n < TN; ++n)
            {
#pragma unroll
                for (unsigned m = 0U; m < TM; ++m)
                    sums[n * TM + m] += dot(regM[m], regN[n]);
            }
        }
        else
        {
#pragma unroll
            for (unsigned m = 0U; m < TM; ++m)
            {
#pragma unroll
                for (unsigned n = 0; n < TN; ++n)
                    sums[m * TN + n] += dot(regM[m], regN[n]);
            }
        }
    }
}

// Store results from registers in matrix C (global memory)
// Each thread store TMxTN elements
template <unsigned SK, unsigned TM, unsigned TN, bool VEC, InputType T, AccumType SumType>
__device__ void storeResult(T *mat, unsigned row, unsigned col, unsigned rows, unsigned cols, const SumType* sums)
{
    constexpr bool isMatSumTypeMatch = std::is_same_v<T, SumType>;
    // Use vectorized memory access only when it is specified and certain conditions are met
    using VecType = typename std::conditional_t<VEC && SK == 1U && isMatSumTypeMatch, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);

#pragma unroll
    for (unsigned m = 0U; m < TM; ++m)
    {
#pragma unroll
        for (unsigned n = 0U; n < TN; n += VEC_SIZE)
        {
            if((row + m) < rows && (col + n) < cols)
            {
                const unsigned idxC = (row + m) * cols + col + n; // index for row major mat C format
                if constexpr (SK == 1U)
                {
                    // Vectorized memory access can be used only when data types match
                    if constexpr (isMatSumTypeMatch)
                        *reinterpret_cast<VecType*>(&mat[idxC]) = *reinterpret_cast<const VecType*>(&sums[m * TN + n]);
                    else
                        mat[idxC] = static_cast<T>(sums[m * TN + n]);
                }
                else // atomic add for 'split K' optimization
                    atomicAdd(&mat[idxC], static_cast<T>(sums[m * TN + n]));
            }
        }
    }
}

// Store results from registers in matrix C (global memory) transposed (CT);
// Sums are transposed as well for better memory access
// Each thread store TMxTN elements
template <unsigned SK, unsigned TM, unsigned TN, bool VEC, InputType T, AccumType SumType>
__device__ void storeResultCT(T *mat, unsigned row, unsigned col, unsigned rows, unsigned cols, const SumType* sums)
{
    constexpr bool isMatSumSameType = std::is_same_v<T, SumType>;
    // Use vectorized memory access only when it is specified and certain conditions are met
    using VecType = typename std::conditional_t<VEC && SK == 1U && isMatSumSameType, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);

#pragma unroll
    for (unsigned n = 0U; n < TN; ++n)
    {
#pragma unroll
        for (unsigned m = 0U; m < TM; m += VEC_SIZE)
        {
            if((row + m) < rows && (col + n) < cols)
            {
                const unsigned idxC = (col + n) * rows + row + m; // index for column major mat C format
                if constexpr (SK == 1U)
                {
                    // Vectorized memory access can be used only when data types match
                    if constexpr (isMatSumSameType)
                        *reinterpret_cast<VecType*>(&mat[idxC]) = *reinterpret_cast<const VecType*>(&sums[n * TM + m]);
                    else
                        mat[idxC] = static_cast<T>(sums[n * TM + m]);
                }
                else // atomic add for 'split K' optimization
                    atomicAdd(&mat[idxC], static_cast<T>(sums[n * TM + m]));
            }
        }
    }
}

///// Kernels /////

// Naive matmul kernel;
// COAL - true for coalescing memory access for row major matrices
// SK - number of K splits
template <unsigned SK, bool COAL, bool AT, bool BT, bool CT, InputType T>
__global__ void matmul_naive_kernel(const T *A, const T *B, T *C, unsigned M, unsigned N, unsigned K)
{
    using SumType = typename std::conditional_t<std::is_same_v<T, __half>, float, T>;

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
        SumType sum = static_cast<T>(0);
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

            sum += dot(valA, valB);
        }

        unsigned idxC;
        if constexpr (CT)
            idxC = col * M + row;
        else
            idxC = row * N + col;

        if constexpr(SK == 1U)
            C[idxC] = static_cast<T>(sum);
        else
            atomicAdd(&C[idxC],  static_cast<T>(sum));
    }
}

// Matmul kernel with tiled memory access: BMxBK - A; BKxBN - B; BMxBN - C
// Each thread process TMxTN elements
template <unsigned SK, unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN, bool AT, bool BT, bool CT,
          bool VEC, InputType T>
__global__ void matmul_tiles_kernel(const T *A, const T *B, T *C, unsigned M, unsigned N, unsigned K)
{
    using SumType = typename std::conditional_t<std::is_same_v<T, __half>, float, T>;
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

    SumType sums[TM * TN] = {static_cast<SumType>(0)};
    T regM[TM] = {static_cast<T>(0)};
    T regN[TN] = {static_cast<T>(0)};

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

        processTiles<BM, BN, BK, TM, TN, STORE_AT, CT, VEC>(ty, tx, smemA, smemB, regM, regN, sums);
        __syncthreads();
    }

    if constexpr (CT)
        storeResultCT<SK, TM, TN, VEC>(C, row, col, M, N, sums);
    else
        storeResult<SK, TM, TN, VEC>(C, row, col, M, N, sums);
}

// Double buffered matmul kernel with tiled memory access: BMxBK - A; BKxBN - B; BMxBN - C
// Each thread process TMxTN elements
template <unsigned SK, unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN, bool AT, bool BT, bool CT,
          bool VEC, InputType T>
__global__ void matmul_tiles_SDBuf_kernel(const T *A, const T *B, T *C, unsigned M, unsigned N, unsigned K)
{
    using SumType = typename std::conditional_t<std::is_same_v<T, __half>, float, T>;
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

    SumType sums[TM * TN] = {static_cast<SumType>(0)};
    T regM[TM] = {static_cast<T>(0)};
    T regN[TN] = {static_cast<T>(0)};

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

    unsigned currentIdx = 0U;
    unsigned nextIdx = 1U;

    loadToShared<BM, BN, BK, NUM_THREADS, AT, BT, STORE_AT, VEC>(A, B, M, N, K, smemA[currentIdx], smemB[currentIdx], threadIdx.x, splitStart);
    __syncthreads();

    for (unsigned blockOffset = splitStart + BK; blockOffset < splitEnd; blockOffset += BK)
    {
        loadToShared<BM, BN, BK, NUM_THREADS, AT, BT, STORE_AT, VEC>(A, B, M, N, K, smemA[nextIdx], smemB[nextIdx], threadIdx.x, blockOffset);

        processTiles<BM, BN, BK, TM, TN, STORE_AT, CT, VEC>(ty, tx, smemA[currentIdx], smemB[currentIdx], regM, regN, sums);
        __syncthreads();

        std::swap(currentIdx, nextIdx);
    }

    processTiles<BM, BN, BK, TM, TN, STORE_AT, CT, VEC>(ty, tx, smemA[currentIdx], smemB[currentIdx], regM, regN, sums);

    if constexpr (CT)
        storeResultCT<SK, TM, TN, VEC>(C, row, col, M, N, sums);
    else
        storeResult<SK, TM, TN, VEC>(C, row, col, M, N, sums);
}

// Double buffered V2 matmul kernel with tiled memory access: BMxBK - A; BKxBN - B; BMxBN - C
// Each thread process TMxTN elements
template <unsigned SK, unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN, bool AT, bool BT, bool CT,
          bool VEC, InputType T>
__global__ void matmul_tiles_DBuf_kernel(const T *A, const T *B, T *C, unsigned M, unsigned N, unsigned K)
{
    using SumType = typename std::conditional_t<std::is_same_v<T, __half>, float, T>;
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

    SumType sums[TM * TN] = {static_cast<SumType>(0)};
    T regM[TM] = {static_cast<T>(0)};
    T regN[TN] = {static_cast<T>(0)};

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

    unsigned doubleBufferIdx = static_cast<unsigned>(threadIdx.x >= NUM_THREADS2);

    if(doubleBufferIdx == 0U)
        loadToShared<BM, BN, BK, NUM_THREADS2, AT, BT, STORE_AT, VEC>(A, B, M, N, K, smemA[0U], smemB[0U], threadIdx.x, splitStart);
    __syncthreads();

    for (unsigned blockOffset = splitStart; blockOffset < splitEnd; blockOffset += 2U * BK)
    {
        if(doubleBufferIdx == 0U)
        {
            processTiles<BM, BN, BK, TM, TN, STORE_AT, CT, VEC>(ty, tx, smemA[0U], smemB[0U], regM, regN, sums);
            __syncthreads();

            if(blockOffset + BK < splitEnd)
                processTiles<BM, BN, BK, TM, TN, STORE_AT, CT, VEC>(ty, tx, smemA[1U], smemB[1U], regM, regN, sums);
            __syncthreads();

            if(blockOffset + 2U * BK < splitEnd)
                loadToShared<BM, BN, BK, NUM_THREADS2, AT, BT, STORE_AT, VEC>(A, B, M, N, K, smemA[0U], smemB[0U], threadIdx.x, blockOffset + 2U * BK);
            __syncthreads();
        }
        else
        {
            if(blockOffset + BK < splitEnd)
                loadToShared<BM, BN, BK, NUM_THREADS2, AT, BT, STORE_AT, VEC>(A, B, M, N, K, smemA[1U], smemB[1U], threadIdx.x - NUM_THREADS2, blockOffset + BK);
            __syncthreads();

            processTiles<BM, BN, BK, TM, TN, STORE_AT, CT, VEC>(ty, tx, smemA[0U], smemB[0U], regM, regN, sums);
            __syncthreads();

            if(blockOffset + BK < splitEnd)
                processTiles<BM, BN, BK, TM, TN, STORE_AT, CT, VEC>(ty, tx, smemA[1U], smemB[1U], regM, regN, sums);
            __syncthreads();
        }
    }

    if constexpr (CT)
        storeResultCT<SK, TM, TN, VEC>(C, row, col, M, N, sums);
    else
        storeResult<SK, TM, TN, VEC>(C, row, col, M, N, sums);
}

// Matmul kernel with tiled memory access: BMxBK - A; BKxBN - B; BMxBN - C
// That utilizes tensor cores (via wmma instructions) for fp16 data type
template <unsigned BM, unsigned BN, unsigned BK, unsigned WM, unsigned WN, unsigned WK, bool AT, bool BT, bool CT, bool VEC>
__global__ void matmul_tiles_wmma_kernel(const __half *A, const __half *B, __half *C, unsigned M, unsigned N, unsigned K)
{
    using SumType = __half;
    using VecType = typename std::conditional_t<VEC, int4, __half>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(__half);
    constexpr bool STORE_AT = true;
    constexpr unsigned NUM_THREADS = BM * BN  * 32U / (WM * WN);

    static_assert(BM % WM == 0U);
    static_assert(BN % WN == 0U);
    static_assert(BK % WK == 0U);
    static_assert(BM * BK % NUM_THREADS == 0U);
    static_assert(BK * BN % NUM_THREADS == 0U);
    static_assert(BM % VEC_SIZE == 0U);
    static_assert(BK % VEC_SIZE == 0U);
    static_assert(BN % VEC_SIZE == 0U);
    static_assert((BM * BK / VEC_SIZE) % NUM_THREADS == 0U);
    static_assert((BN * BK / VEC_SIZE) % NUM_THREADS == 0U);

    const unsigned wIdx = threadIdx.x / 32;
    const unsigned ty = wIdx / (BN / WN) * WM;
    const unsigned tx = wIdx % (BN / WN) * WN;

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

    const unsigned row = by * BM + ty;
    const unsigned col = bx * BN + tx;

    __shared__ __half smemA[BK * BM];  // STORE_AT - transposed or not; size always BK * BM
    __shared__ __half smemB[BK * BN];

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WN, WK, SumType> sums;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WM, WN, WK, __half, nvcuda::wmma::col_major> regM;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WM, WN, WK, __half, nvcuda::wmma::row_major> regN;

    nvcuda::wmma::fill_fragment(sums, static_cast<__half>(0));

    for (unsigned blockOffset = 0U; blockOffset < K; blockOffset += BK)
    {
        loadToShared<BM, BN, BK, NUM_THREADS, AT, BT, STORE_AT, VEC>(A, B, M, N, K, smemA, smemB, threadIdx.x, blockOffset);
        __syncthreads();

        for(unsigned k = 0U; k < BK; k += WK)
        {
            nvcuda::wmma::load_matrix_sync(regM, &smemA[k * BM + ty], BM);

            nvcuda::wmma::load_matrix_sync(regN, &smemB[k * BN + tx], BN);

            nvcuda::wmma::mma_sync(sums, regM, regN, sums);
        }
        __syncthreads();
    }

    if(row < M && col < N)
    {
        if constexpr (CT)
            nvcuda::wmma::store_matrix_sync(&C[col * M + row], sums, M, nvcuda::wmma::mem_col_major);
        else
            nvcuda::wmma::store_matrix_sync(&C[row * N + col], sums, N, nvcuda::wmma::mem_row_major);  
    }
}

///// Special Kernels /////

// Matmul kernel with row-wise bias vector addition
template <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN, bool AT, bool BT, bool CT,
          bool VEC, InputType T>
__global__ void matmul_bias_tiles_kernel(const T *A, const T *B, const T *bias, T *C, unsigned M, unsigned N, unsigned K)
{
    using SumType = typename std::conditional_t<std::is_same_v<T, __half>, float, T>;
    using VecType = typename std::conditional_t<VEC, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);
    constexpr bool STORE_AT = TM > 1U;
    constexpr bool SK = 1U; // No split K optimization support yer
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
    __shared__ T smemBias[BN]; // Store bias in blocks in shared memory

    T regM[TM] = {static_cast<T>(0)};
    T regN[TN] = {static_cast<T>(0)};
    SumType sums[TM * TN] = {static_cast<SumType>(0)};

    // Load bias to shared memory
    unsigned biasTx = threadIdx.x * VEC_SIZE;
    if(biasTx < BN)
    {
        VecType tmpVal;
        if constexpr(VEC)
            tmpVal = {0, 0, 0, 0};
        else
            tmpVal = static_cast<T>(0);
        unsigned bCol = bx * BN + biasTx;
        if(bCol < N)
            tmpVal = *reinterpret_cast<const VecType*>(&bias[bCol]);
        *reinterpret_cast<VecType*>(&smemBias[biasTx]) = tmpVal;
    }

    // Main matmul loop
    for (unsigned blockOffset = 0U; blockOffset < K; blockOffset += BK)
    {
        loadToShared<BM, BN, BK, NUM_THREADS, AT, BT, STORE_AT, VEC>(A, B, M, N, K, smemA, smemB, threadIdx.x, blockOffset);
        __syncthreads();

        processTiles<BM, BN, BK, TM, TN, STORE_AT, CT, VEC>(ty, tx, smemA, smemB, regM, regN, sums);
        __syncthreads();
    }

    // Add bias
    T regBias[TN];

#pragma unroll
    for (unsigned n = 0U; n < TN; n += VEC_SIZE)
        *reinterpret_cast<VecType*>(&regBias[n]) = *reinterpret_cast<const VecType*>(&smemBias[tx + n]);

#pragma unroll
    for (unsigned m = 0U; m < TM; ++m)
    {
#pragma unroll
        for (unsigned n = 0U; n < TN; ++n)
        {
            if constexpr (CT)
                sums[n * TM + m] += static_cast<SumType>(regBias[n]);
            else
                sums[m * TN + n] += static_cast<SumType>(regBias[n]);
        }
    }

    // Store results
    if constexpr (CT)
        storeResultCT<SK, TM, TN, VEC>(C, row, col, M, N, sums);
    else
        storeResult<SK, TM, TN, VEC>(C, row, col, M, N, sums);
}

///// Launch functions /////

using RuntimeBool = std::variant<std::bool_constant<false>, std::bool_constant<true>>;

inline RuntimeBool to_variant(bool val) {
    return val ? RuntimeBool{std::bool_constant<true>{}} : RuntimeBool{std::bool_constant<false>{}};
}

template <unsigned blockDimX = 16U, unsigned blockDimY = 16U, bool COAL = true, unsigned SK = 1U, InputType T>
void launch_matmul_naive(const T *A, const T *B, T *C, unsigned M, unsigned N, unsigned K,
                         bool AT, bool BT, bool CT)
{
    const dim3 blockDim(blockDimX, blockDimY);
    const dim3 gridDim(CEIL_DIV(COAL ? N : M, blockDim.x), CEIL_DIV(COAL ? M : N, blockDim.y), SK);

    auto launchKernel = [&](auto b1, auto b2, auto b3) {
        matmul_naive_kernel<SK, COAL, b1.value, b2.value, b3.value><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    };

    std::visit(launchKernel, to_variant(AT), to_variant(BT), to_variant(CT));
}

template <unsigned BM = 128U, unsigned BN = 128U, unsigned BK = 8U, unsigned TM = 8U, unsigned TN = 8U,
          bool VEC = true, bool DBUF = true, unsigned SK = 1U, InputType T>
void launch_matmul_tiles(const T *A, const T *B, T *C, unsigned M, unsigned N, unsigned K,
                         bool AT, bool BT, bool CT)
{
    using VecType = typename std::conditional_t<VEC, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);

    const dim3 blockDim(BM * BN / (TM * TN));
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), SK);

    auto launchKernel = [&](auto b1, auto b2, auto b3, auto v) {
        if constexpr (DBUF)
            matmul_tiles_DBuf_kernel<SK, BM, BN, BK, TM, TN, b1.value, b2.value, b3.value, VEC && v.value><<<gridDim, blockDim>>>(A, B, C, M, N, K);
        else
            matmul_tiles_kernel<SK, BM, BN, BK, TM, TN, b1.value, b2.value, b3.value, VEC && v.value><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    };

    const unsigned lda = AT ? M : K;
    const unsigned ldb = BT ? K : N;
    const unsigned ldc = CT ? M : N;
    const bool possibleVecAccess = (lda % VEC_SIZE == 0U) && (ldb % VEC_SIZE == 0U) && (ldc % VEC_SIZE == 0U);

    std::visit(launchKernel, to_variant(AT), to_variant(BT), to_variant(CT), to_variant(possibleVecAccess));
}

template <unsigned BM = 64U, unsigned BN = 64U, unsigned BK = 16U, unsigned WM = 16U, unsigned WN = 16U, unsigned WK = 16U,
          bool VEC = true>
void launch_matmul_tiles_wmma(const __half *A, const __half *B, __half *C, unsigned M, unsigned N, unsigned K,
                              bool AT, bool BT, bool CT)
{
    if(M % WM == 0U && N % WN == 0U)
    {
        using VecType = typename std::conditional_t<VEC, int4, __half>;

        constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(__half);

        const dim3 blockDim(BM * BN * 32U / (WN * WM));
        const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

        auto launchKernel = [&](auto b1, auto b2, auto b3, auto v) {
            matmul_tiles_wmma_kernel<BM, BN, BK, WM, WN, WK, b1.value, b2.value, b3.value, VEC && v.value><<<gridDim, blockDim>>>(A, B, C, M, N, K);
        };

        const unsigned lda = AT ? M : K;
        const unsigned ldb = BT ? K : N;
        const unsigned ldc = CT ? M : N;
        const bool possibleVecAccess = (lda % VEC_SIZE == 0U) && (ldb % VEC_SIZE == 0U) && (ldc % VEC_SIZE == 0U);

        std::visit(launchKernel, to_variant(AT), to_variant(BT), to_variant(CT), to_variant(possibleVecAccess));
    }
    else
    {
        launch_matmul_tiles<128U, 128U, 16U, 8U, 8U, VEC, false, 1U>(A, B, C, M, N, K, AT, BT, CT);
    }
}

template <unsigned BM = 128U, unsigned BN = 128U, unsigned BK = 8U, unsigned TM = 8U, unsigned TN = 8U,
          bool VEC = true, InputType T>
void launch_matmul_bias_tiles(const T *A, const T *B, const T* bias, T *C, unsigned M, unsigned N, unsigned K,
                         bool AT, bool BT, bool CT)
{
    using VecType = typename std::conditional_t<VEC, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);

    const dim3 blockDim(BM * BN / (TM * TN));
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1U);

    auto launchKernel = [&](auto b1, auto b2, auto b3, auto v) {
        matmul_bias_tiles_kernel<BM, BN, BK, TM, TN, b1.value, b2.value, b3.value, VEC && v.value><<<gridDim, blockDim>>>(A, B, bias, C, M, N, K);
    };

    const unsigned lda = AT ? M : K;
    const unsigned ldb = BT ? K : N;
    const unsigned ldc = CT ? M : N;
    const bool possibleVecAccess = (lda % VEC_SIZE == 0U) && (ldb % VEC_SIZE == 0U) && (ldc % VEC_SIZE == 0U) && (N % VEC_SIZE == 0U);

    std::visit(launchKernel, to_variant(AT), to_variant(BT), to_variant(CT), to_variant(possibleVecAccess));
}

#endif // MATMUL_KERNELS
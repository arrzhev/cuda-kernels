#ifndef MATMUL_KERNELS
#define MATMUL_KERNELS

#include <variant>
#include <util.cuh>
#include <util_matmul.cuh>
#include <cuda_fp16.h>
#include <mma.h>

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
    constexpr bool ReLU = false;

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
        loadToShared<BM, BN, BK, NUM_THREADS, AT, BT, STORE_AT, VEC, ReLU>(A, static_cast<T*>(nullptr), B, M, N, K, smemA, smemB, threadIdx.x, blockOffset);
        __syncthreads();

        processTiles<BM, BN, BK, TM, TN, STORE_AT, CT, VEC>(ty, tx, smemA, smemB, regM, regN, sums);
        __syncthreads();
    }

    if constexpr (CT)
        storeResultT<SK, TM, TN, VEC>(C, row, col, M, N, sums);
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
    constexpr bool ReLU = false;

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

    loadToShared<BM, BN, BK, NUM_THREADS, AT, BT, STORE_AT, VEC, ReLU>(A, static_cast<T*>(nullptr), B, M, N, K, smemA[currentIdx], smemB[currentIdx], threadIdx.x, splitStart);
    __syncthreads();

    for (unsigned blockOffset = splitStart + BK; blockOffset < splitEnd; blockOffset += BK)
    {
        loadToShared<BM, BN, BK, NUM_THREADS, AT, BT, STORE_AT, VEC, ReLU>(A, static_cast<T*>(nullptr), B, M, N, K, smemA[nextIdx], smemB[nextIdx], threadIdx.x, blockOffset);

        processTiles<BM, BN, BK, TM, TN, STORE_AT, CT, VEC>(ty, tx, smemA[currentIdx], smemB[currentIdx], regM, regN, sums);
        __syncthreads();

        std::swap(currentIdx, nextIdx);
    }

    processTiles<BM, BN, BK, TM, TN, STORE_AT, CT, VEC>(ty, tx, smemA[currentIdx], smemB[currentIdx], regM, regN, sums);

    if constexpr (CT)
        storeResultT<SK, TM, TN, VEC>(C, row, col, M, N, sums);
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
    constexpr bool ReLU = false;

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
        loadToShared<BM, BN, BK, NUM_THREADS2, AT, BT, STORE_AT, VEC, ReLU>(A, static_cast<T*>(nullptr), B, M, N, K, smemA[0U], smemB[0U], threadIdx.x, splitStart);
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
                loadToShared<BM, BN, BK, NUM_THREADS2, AT, BT, STORE_AT, VEC, ReLU>(A, static_cast<T*>(nullptr), B, M, N, K, smemA[0U], smemB[0U], threadIdx.x, blockOffset + 2U * BK);
            __syncthreads();
        }
        else
        {
            if(blockOffset + BK < splitEnd)
                loadToShared<BM, BN, BK, NUM_THREADS2, AT, BT, STORE_AT, VEC, ReLU>(A, static_cast<T*>(nullptr), B, M, N, K, smemA[1U], smemB[1U], threadIdx.x - NUM_THREADS2, blockOffset + BK);
            __syncthreads();

            processTiles<BM, BN, BK, TM, TN, STORE_AT, CT, VEC>(ty, tx, smemA[0U], smemB[0U], regM, regN, sums);
            __syncthreads();

            if(blockOffset + BK < splitEnd)
                processTiles<BM, BN, BK, TM, TN, STORE_AT, CT, VEC>(ty, tx, smemA[1U], smemB[1U], regM, regN, sums);
            __syncthreads();
        }
    }

    if constexpr (CT)
        storeResultT<SK, TM, TN, VEC>(C, row, col, M, N, sums);
    else
        storeResult<SK, TM, TN, VEC>(C, row, col, M, N, sums);
}

// Matmul kernel with tiled memory access: BMxBK - A; BKxBN - B; BMxBN - C
// That utilizes tensor cores (via wmma instructions) for fp16 data type
template <unsigned BM, unsigned BN, unsigned BK, unsigned WM, unsigned WN, unsigned WK, bool AT, bool BT, bool CT, bool VEC>
__global__ void matmul_tiles_wmma_kernel(const __half *A, const __half *B, __half *C, unsigned M, unsigned N, unsigned K)
{
    using SumType = float;
    using VecType = typename std::conditional_t<VEC, int4, __half>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(__half);
    constexpr bool STORE_AT = true;
    constexpr unsigned NUM_THREADS = BM * BN  * 32U / (WM * WN);
    constexpr bool ReLU = false;

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

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WM, WN, WK, __half, nvcuda::wmma::col_major> regM;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WM, WN, WK, __half, nvcuda::wmma::row_major> regN;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WN, WK, __half> regC;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WN, WK, SumType> sums;

    nvcuda::wmma::fill_fragment(sums, static_cast<__half>(0));

    for (unsigned blockOffset = 0U; blockOffset < K; blockOffset += BK)
    {
        loadToShared<BM, BN, BK, NUM_THREADS, AT, BT, STORE_AT, VEC, ReLU>(A, static_cast<__half*>(nullptr), B, M, N, K, smemA, smemB, threadIdx.x, blockOffset);
        __syncthreads();

        for(unsigned k = 0U; k < BK; k += WK)
        {
            nvcuda::wmma::load_matrix_sync(regM, &smemA[k * BM + ty], BM);

            nvcuda::wmma::load_matrix_sync(regN, &smemB[k * BN + tx], BN);

            nvcuda::wmma::mma_sync(sums, regM, regN, sums);
        }
        __syncthreads();
    }

    // Store fp32 accumulator in fp16
    for(int i = 0; i < sums.num_elements; i++)
      regC.x[i] = __float2half(sums.x[i]);

    if(row < M && col < N)
    {
        if constexpr (CT)
            nvcuda::wmma::store_matrix_sync(&C[col * M + row], regC, M, nvcuda::wmma::mem_col_major);
        else
            nvcuda::wmma::store_matrix_sync(&C[row * N + col], regC, N, nvcuda::wmma::mem_row_major);  
    }
}


///// Launch functions /////

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

#endif // MATMUL_KERNELS
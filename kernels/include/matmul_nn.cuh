#ifndef MATMUL_NN_KERNELS
#define MATMUL_NN_KERNELS

#include <variant>
#include <util.cuh>
#include <util_matmul.cuh>
#include <cuda_fp16.h>

///// Kernels /////

// Matmul kernel with row-wise bias vector addition
template <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN, bool AT, bool BT, bool CT,
          bool VEC, bool ReLU, InputType T>
__global__ void matmul_bias_tiles_kernel(const T *A, const T *B, const T *bias, T *C, T *CR, unsigned M, unsigned N, unsigned K)
{
    using SumType = typename std::conditional_t<std::is_same_v<T, __half>, float, T>;
    using VecType = typename std::conditional_t<VEC, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);
    constexpr bool STORE_AT = TM > 1U;
    constexpr bool SK = 1U; // No split K optimization support yer
    constexpr unsigned NUM_THREADS = BM * BN / (TM * TN);
    constexpr bool LOAD_ReLU = false;

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
        loadToShared<BM, BN, BK, NUM_THREADS, AT, BT, STORE_AT, VEC, LOAD_ReLU>(A, static_cast<T*>(nullptr), B, M, N, K, smemA, smemB, threadIdx.x, blockOffset);
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
        storeResultT<SK, TM, TN, VEC>(C, row, col, M, N, sums);
    else
        storeResult<SK, TM, TN, VEC>(C, row, col, M, N, sums);

    // Handle fused ReLU activation
    if constexpr (ReLU)
    {
        constexpr SumType minValue = static_cast<SumType>(0);
#pragma unroll
        for (unsigned m = 0U; m < TM; ++m)
        {
#pragma unroll
            for (unsigned n = 0U; n < TN; ++n)
            {
                unsigned idxC;
                if constexpr (CT)
                    idxC = n * TM + m;
                else
                    idxC = m * TN + n;

                if(sums[idxC] < minValue)
                    sums[idxC] = minValue;
            }
        }

        if constexpr (CT)
            storeResultT<SK, TM, TN, VEC>(CR, row, col, M, N, sums);
        else
            storeResult<SK, TM, TN, VEC>(CR, row, col, M, N, sums);
    }
}

// Matrix multiplication fused with relu backward step for matrix A
template <unsigned SK, unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN, bool AT, bool BT, bool CT,
          bool VEC, InputType T>
__global__ void matmul_tiles_relu_kernel(const T *A, const T *AR, const T *B, T *C, unsigned M, unsigned N, unsigned K)
{
    using SumType = typename std::conditional_t<std::is_same_v<T, __half>, float, T>;
    using VecType = typename std::conditional_t<VEC, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);
    constexpr bool STORE_AT = TM > 1U;
    constexpr unsigned NUM_THREADS = BM * BN / (TM * TN);
    constexpr bool ReLU = true;

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
        loadToShared<BM, BN, BK, NUM_THREADS, AT, BT, STORE_AT, VEC, ReLU>(A, AR, B, M, N, K, smemA, smemB, threadIdx.x, blockOffset);
        __syncthreads();

        processTiles<BM, BN, BK, TM, TN, STORE_AT, CT, VEC>(ty, tx, smemA, smemB, regM, regN, sums);
        __syncthreads();
    }

    if constexpr (CT)
        storeResultT<SK, TM, TN, VEC>(C, row, col, M, N, sums);
    else
        storeResult<SK, TM, TN, VEC>(C, row, col, M, N, sums);
}

///// Launch functions /////

template <unsigned BM = 128U, unsigned BN = 128U, unsigned BK = 8U, unsigned TM = 8U, unsigned TN = 8U,
          bool VEC = true, bool ReLU, InputType T>
void launch_matmul_bias_tiles(const T *A, const T *B, const T* bias, T *C, T *CR, unsigned M, unsigned N, unsigned K,
                         bool AT, bool BT, bool CT)
{
    using VecType = typename std::conditional_t<VEC, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);

    const dim3 blockDim(BM * BN / (TM * TN));
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1U);

    auto launchKernel = [&](auto b1, auto b2, auto b3, auto v) {
        matmul_bias_tiles_kernel<BM, BN, BK, TM, TN, b1.value, b2.value, b3.value, VEC && v.value, ReLU><<<gridDim, blockDim>>>(A, B, bias, C, CR, M, N, K);
    };

    const unsigned lda = AT ? M : K;
    const unsigned ldb = BT ? K : N;
    const unsigned ldc = CT ? M : N;
    const bool possibleVecAccess = (lda % VEC_SIZE == 0U) && (ldb % VEC_SIZE == 0U) && (ldc % VEC_SIZE == 0U) && (N % VEC_SIZE == 0U);

    std::visit(launchKernel, to_variant(AT), to_variant(BT), to_variant(CT), to_variant(possibleVecAccess));
}

template <unsigned BM = 128U, unsigned BN = 128U, unsigned BK = 8U, unsigned TM = 8U, unsigned TN = 8U,
          bool VEC = true, unsigned SK = 1U, InputType T>
void launch_matmul_relu_tiles(const T *A, const T *AR, const T *B, T *C, unsigned M, unsigned N, unsigned K,
                         bool AT, bool BT, bool CT)
{
    using VecType = typename std::conditional_t<VEC, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);

    const dim3 blockDim(BM * BN / (TM * TN));
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), SK);

    auto launchKernel = [&](auto b1, auto b2, auto b3, auto v) {
            matmul_tiles_relu_kernel<SK, BM, BN, BK, TM, TN, b1.value, b2.value, b3.value, VEC && v.value><<<gridDim, blockDim>>>(A, AR, B, C, M, N, K);
    };

    const unsigned lda = AT ? M : K;
    const unsigned ldb = BT ? K : N;
    const unsigned ldc = CT ? M : N;
    const bool possibleVecAccess = (lda % VEC_SIZE == 0U) && (ldb % VEC_SIZE == 0U) && (ldc % VEC_SIZE == 0U);

    std::visit(launchKernel, to_variant(AT), to_variant(BT), to_variant(CT), to_variant(possibleVecAccess));
}

#endif // MATMUL_NN_KERNELS
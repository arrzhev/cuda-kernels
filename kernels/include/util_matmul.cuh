#ifndef UTIL_MATMUL_KERNELS
#define UTIL_MATMUL_KERNELS

#include <concepts>
#include <util.cuh>
#include <cuda_fp16.h>

// Type of accumulator to store dot product of each element
template <typename T>
concept AccumType = IsAnyOf<T, int, double, float>;

// Load blocks (BMxBK and BKxBN) of A, B matrices from global memory to shared memory
// AT, BT - specify layout of A, B matrices; true - column major, false - row major
// STOR_AT - indicate if shared memory block for matrix A is transposed
// VEC - indicate if vectorized memory access allowed
template <unsigned BM, unsigned BN, unsigned BK, unsigned NUM_THREADS,
          bool AT, bool BT, bool STORE_AT, bool VEC, bool ReLU, InputType T>
__device__ void loadToShared(const T *A, const T *AR, const T *B, unsigned M, unsigned N, unsigned K,
    T* smemA, T* smemB, unsigned threadIdx, unsigned blockOffset)
{
    using VecType = typename std::conditional_t<VEC, int4, T>;

    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);
    const T minValue = static_cast<T>(0);

    // Load block (B_ROWS x B_COLS) of row-major matrix (A or B) from global memory to shared memory
    auto loadMatToShared = [&] <unsigned B_ROWS, unsigned B_COLS, bool STORE_T, bool LOAD_ReLU> (
        const T *mat, const T *matR, unsigned rows, unsigned cols, T* smem, unsigned rowOffset, unsigned colOffset)
    {
        constexpr unsigned B_COLS_4 = B_COLS / VEC_SIZE;

#pragma unroll
        for (unsigned threadOffset = 0U; threadOffset < B_ROWS * B_COLS / VEC_SIZE; threadOffset += NUM_THREADS)
        {
            unsigned rowTile = (threadIdx + threadOffset) / B_COLS_4;
            unsigned colTile = (threadIdx + threadOffset) % B_COLS_4 * VEC_SIZE;
            const unsigned row = rowOffset + rowTile;
            const unsigned col = colOffset + colTile;

            VecType tmpVal;
            if constexpr(VEC)
                tmpVal = {0, 0, 0, 0};
            else
                tmpVal = static_cast<VecType>(0);

            if(row < rows && col < cols)
            {
                const unsigned idx = row * cols + col;
                tmpVal =  *reinterpret_cast<const VecType*>(&mat[idx]);
                if constexpr (LOAD_ReLU)
                {
                    VecType tmpValR = *reinterpret_cast<const VecType*>(&matR[idx]);
#pragma unroll
                    for(unsigned i = 0U; i < VEC_SIZE; ++i)
                    {
                        if(reinterpret_cast<T*>(&tmpValR)[i] <= minValue)
                            reinterpret_cast<T*>(&tmpVal)[i] = minValue;
                    }
                }
            }

            if constexpr(STORE_T)
            {
#pragma unroll
                for(unsigned i = 0U; i < VEC_SIZE; ++i)
                    smem[(colTile + i) * B_ROWS + rowTile] = reinterpret_cast<T*>(&tmpVal)[i];
            }
            else
                *reinterpret_cast<VecType*>(&smem[rowTile * B_COLS + colTile]) = tmpVal;  

        }
    };

    // Load block (B_ROWS x B_COLS) of column-major matrix (A or B) from global memory to shared memory
    auto loadMatToSharedT = [&] <unsigned B_ROWS, unsigned B_COLS, bool STORE_T, bool LOAD_ReLU> (
        const T *mat, const T *matR, unsigned rows, unsigned cols, T* smem, unsigned rowOffset, unsigned colOffset)
    {
        constexpr unsigned B_ROWS_4 = B_ROWS / VEC_SIZE;

#pragma unroll
        for (unsigned threadOffset = 0U; threadOffset < B_ROWS * B_COLS / VEC_SIZE; threadOffset += NUM_THREADS)
        {
            unsigned rowTile = (threadIdx + threadOffset) % B_ROWS_4 * VEC_SIZE;
            unsigned colTile = (threadIdx + threadOffset) / B_ROWS_4;
            const unsigned row = rowOffset + rowTile;
            const unsigned col = colOffset + colTile;

            VecType tmpVal;
            if constexpr(VEC)
                tmpVal = {0, 0, 0, 0};
            else
                tmpVal = static_cast<VecType>(0);

            if(row < rows && col < cols)
            {
                const unsigned idx = col * rows + row;
                tmpVal =  *reinterpret_cast<const VecType*>(&mat[idx]);
                if constexpr (LOAD_ReLU)
                {
                    VecType tmpValR = *reinterpret_cast<const VecType*>(&matR[idx]);
#pragma unroll
                    for(unsigned i = 0U; i < VEC_SIZE; ++i)
                    {
                        if(reinterpret_cast<T*>(&tmpValR)[i] <= minValue)
                            reinterpret_cast<T*>(&tmpVal)[i] = minValue;
                    }
                }
            }

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
    };

    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;

    // Load block (BM X BK) of matrix A[by*BM][blockOffset] from global memory to shared
    if constexpr (AT)
        loadMatToSharedT.template operator()<BM, BK, STORE_AT, ReLU>(A, AR, M, K, smemA, by * BM, blockOffset);
    else
        loadMatToShared.template operator()<BM, BK, STORE_AT, ReLU>(A, AR, M, K, smemA, by * BM, blockOffset);

    // Load block (BK X BN) of matrix N[blockOffset][bx*BN] from global memory to shared
    if constexpr (BT)
        loadMatToSharedT.template operator()<BK, BN, false, false>(B, static_cast<T*>(nullptr), K, N, smemB, blockOffset, bx * BN);
    else
        loadMatToShared.template operator()<BK, BN, false, false>(B, static_cast<T*>(nullptr), K, N, smemB, blockOffset, bx * BN);
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

// Store results from registers in transposed matrix C (global memory);
// Sums should be transposed as well for better memory access
// Each thread store TMxTN elements
template <unsigned SK, unsigned TM, unsigned TN, bool VEC, InputType T, AccumType SumType>
__device__ void storeResultT(T *mat, unsigned row, unsigned col, unsigned rows, unsigned cols, const SumType* sums)
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

#endif //nUTIL_MATMUL_KERNELS
#ifndef MATRIX_REDUCTION_KERNELS
#define MATRIX_REDUCTION_KERNELS

#include <util.cuh>

///// Kernels /////

template <bool VEC, InputType T>
__global__ void matrixRowReduction_naive_kernel(const T *A, T *B, unsigned rows, unsigned cols)
 {
    using SumType = typename std::conditional_t<std::is_same_v<T, __half>, float, T>;
    using VecType = typename std::conditional_t<VEC, int4, T>;
    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);

    const unsigned col = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;

    if (col < cols)
    {
        SumType sum[VEC_SIZE] = {static_cast<SumType>(0)};

        for (unsigned i = 0U; i < rows; ++i)
        {
            const VecType valA = *reinterpret_cast<const VecType*>(&A[i * cols + col]);
#pragma unroll
            for(unsigned s = 0U; s < VEC_SIZE; ++s)
                sum[s] += static_cast<SumType>(reinterpret_cast<const T*>(&valA)[s]);
        }

        T reg[VEC_SIZE];
#pragma unroll
        for(unsigned s = 0U; s < VEC_SIZE; ++s)
            reg[s] = static_cast<T>(sum[s]);

        *reinterpret_cast<VecType*>(&B[col]) = *reinterpret_cast<VecType*>(&reg);
    }
}

template <bool VEC, InputType T>
__global__ void matrixRowReduction_kernel(const T *A, T *B, unsigned rows, unsigned cols)
 {
    using SumType = typename std::conditional_t<std::is_same_v<T, __half>, float, T>;
    using VecType = typename std::conditional_t<VEC, int4, T>;
    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);

    const unsigned col = blockIdx.x * VEC_SIZE;
    const unsigned row = threadIdx.x;
    extern __shared__ SumType smem[];

    SumType sum[VEC_SIZE] = {static_cast<SumType>(0)};

    for (unsigned i = row; i < rows; i+= blockDim.x)
    {
        const VecType valA = *reinterpret_cast<const VecType*>(&A[i * cols + col]);
#pragma unroll
        for(unsigned s = 0U; s < VEC_SIZE; ++s)
            sum[s] += static_cast<SumType>(reinterpret_cast<const T*>(&valA)[s]);

    }

#pragma unroll
    for(unsigned s = 0U; s < VEC_SIZE; ++s)
        smem[s * blockDim.x + row] = sum[s];
    __syncthreads();

#pragma unroll
    for(unsigned s = 0U; s < VEC_SIZE; ++s)
    {
        for (unsigned stride = blockDim.x>>1U; stride > 0U; stride>>=1U)
        {
            if (row < stride)
                smem[s * blockDim.x + row] += smem[s * blockDim.x + row + stride];
            __syncthreads();
        }
    }

    if (row == 0U)
    {
        T reg[VEC_SIZE];
#pragma unroll
        for(unsigned s = 0U; s < VEC_SIZE; ++s)
            reg[s] = static_cast<T>(smem[s * blockDim.x]);

        *reinterpret_cast<VecType*>(&B[col]) = *reinterpret_cast<VecType*>(&reg);
    }
}

///// Launch functions /////


template <unsigned blockDim, InputType T>
void launch_matrixRowReduction_naive(const T *A, T *B, unsigned rows, unsigned cols)
{
    constexpr unsigned VEC_SIZE = sizeof(int4) / sizeof(T);
    // If input matrix is VEC_SIZE aligned - use vectorized memory access
    if(cols % VEC_SIZE == 0U)
    {
        const unsigned gridDim = CEIL_DIV(cols / VEC_SIZE, blockDim);
        matrixRowReduction_naive_kernel<true><<<gridDim, blockDim>>>(A, B, rows, cols);
    }
    else
    {
        const unsigned gridDim = CEIL_DIV(cols, blockDim);
        matrixRowReduction_naive_kernel<false><<<gridDim, blockDim>>>(A, B, rows, cols);
    }
}

template <unsigned blockDim, InputType T>
void launch_matrixRowReduction(const T *A, T *B, unsigned rows, unsigned cols)
{
    constexpr unsigned VEC_SIZE = sizeof(int4) / sizeof(T);
    // If input matrix is VEC_SIZE aligned - use vectorized memory access
    if(cols % VEC_SIZE == 0U)
    {
        const unsigned gridDim = cols / VEC_SIZE;
        const size_t sharedSize = VEC_SIZE * blockDim * sizeof(float);
        matrixRowReduction_kernel<true><<<gridDim, blockDim, sharedSize>>>(A, B, rows, cols);
    }
    else
    {
        const unsigned gridDim = cols;
        const size_t sharedSize = blockDim * sizeof(float);
        matrixRowReduction_kernel<false><<<gridDim, blockDim, sharedSize>>>(A, B, rows, cols);
    }
}

#endif // MATRIX_REDUCTION_KERNELS
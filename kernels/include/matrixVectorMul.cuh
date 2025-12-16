#ifndef MATRIX_VECTOR_MUL_KERNELS
#define MATRIX_VECTOR_MUL_KERNELS

#include <concepts>
#include <util.cuh>

// Input data format
template <typename T>
concept floatDim = IsAnyOf<T, float, float4>;

///// Kernels /////

// Naive matrix x vector multiplication kernel
template <floatDim T>
__global__ void matrixVectorMul_naive_kernel(const T *X, const T *y, float *z, unsigned rows, unsigned cols)
 {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        float sum = 0.0f;
        X += (row * cols);

        for (unsigned i = 0U; i < cols; ++i)
            sum += dot(X[i], y[i]);

        z[row] = sum;
    }
}

// Matrix x vector multiplication kernel with shared memory optimization
template <floatDim T>
__global__ void matrixVectorMul_shared_kernel(const T *X, const T *y, float *z, unsigned rows, unsigned cols)
 {
    const unsigned row = blockIdx.x;
    const unsigned col = threadIdx.x;
    extern __shared__ float smem[];

    float sum = 0.0f;

    X += (row * cols);

    for (unsigned i = col; i < cols; i+= blockDim.x)
        sum += dot(X[i], y[i]);

    smem[col] = sum;
    __syncthreads();

    for (unsigned stride = blockDim.x>>1U; stride > 0U; stride>>=1U)
    {
        if (col < stride)
            smem[col] += smem[col + stride];
        __syncthreads();
    }

    if(col == 0U)
        z[row] = smem[0U];
}

// Matrix x vector multiplication kernel with block and warp reduction optimization
template <floatDim T>
__global__ void matrixVectorMul_warp_kernel(const T *X, const T *y, float *z, unsigned rows, unsigned cols)
{
    const unsigned row = blockIdx.x;
    const unsigned col = threadIdx.x;

    float sum = 0.0f;

    X += (row * cols);

    for (unsigned i = col; i < cols; i+= blockDim.x)
        sum += dot(X[i], y[i]);

    sum = blockReduceSum(sum);

    if (col == 0U)
        z[row] = sum;
}

///// Launch functions /////

template <unsigned blockDim>
void launch_matrixVectorMul_naive(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
{
    const unsigned gridDim = CEIL_DIV(rows, blockDim);

    // If size is a multiply of 4 - use vectorized memory access
    if(cols % 4U == 0U)
    {
        cols /= 4U;
        matrixVectorMul_naive_kernel<<<gridDim, blockDim>>>(reinterpret_cast<const float4*>(X),
         reinterpret_cast<const float4*>(y), z, rows, cols);
    }
    else
        matrixVectorMul_naive_kernel<<<gridDim, blockDim>>>(X, y, z, rows, cols);
}

template <unsigned blockDim>
void launch_matrixVectorMul_shared(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
{
    const unsigned gridDim = rows;
    const size_t sharedSize = blockDim * sizeof(float);

    // If size is a multiply of 4 - use vectorized memory access
    if(cols % 4U == 0U)
    {
        cols /= 4U;
        matrixVectorMul_shared_kernel<<<gridDim, blockDim, sharedSize>>>(reinterpret_cast<const float4*>(X),
         reinterpret_cast<const float4*>(y), z, rows, cols);
    }
    else
        matrixVectorMul_shared_kernel<<<gridDim, blockDim, sharedSize>>>(X, y, z, rows, cols);
}

template <unsigned blockDim>
void launch_matrixVectorMul_warp(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
{
    const unsigned gridDim = rows;

    // If size is a multiply of 4 - use vectorized memory access
    if(cols % 4U == 0U)
    {
        cols /= 4U;
        matrixVectorMul_warp_kernel<<<gridDim, blockDim>>>(reinterpret_cast<const float4*>(X),
         reinterpret_cast<const float4*>(y), z, rows, cols);
    }
    else
        matrixVectorMul_warp_kernel<<<gridDim, blockDim>>>(X, y, z, rows, cols);
}

#endif // MATRIX_VECTOR_MUL_KERNELS
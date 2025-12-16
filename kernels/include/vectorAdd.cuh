#ifndef VECTOR_ADD_KERNELS
#define VECTOR_ADD_KERNELS

#include <concepts>
#include <util.cuh>

// Input data format
template <typename T>
concept floatDim = IsAnyOf<T, float, float4>;

///// Kernels /////

template <floatDim T>
__global__ void vectorAdd_kernel(const T *x, const T *y, T *z, unsigned size)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        z[i] = x[i] + y[i];
}

///// Launch functions /////

template <unsigned blockDim>
void launch_vectorAdd(const float *x, const float *y, float *z, unsigned size)
{
    // If size is a multiply of 4 - use vectorized memory access
    if(size % 4U == 0U)
    {
        size /= 4U;
        const unsigned gridDim = CEIL_DIV(size, blockDim);
        vectorAdd_kernel<<<gridDim, blockDim>>>(reinterpret_cast<const float4*>(x), reinterpret_cast<const float4*>(y),
            reinterpret_cast<float4*>(z), size);
    }
    else
    {
        const unsigned gridDim = CEIL_DIV(size, blockDim);
        vectorAdd_kernel<<<gridDim, blockDim>>>(x, y, z, size);
    }
}

#endif // VECTOR_ADD_KERNELS
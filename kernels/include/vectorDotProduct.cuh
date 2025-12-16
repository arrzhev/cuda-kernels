#ifndef VECTOR_DOT_PRODUCT_KERNELS
#define VECTOR_DOT_PRODUCT_KERNELS

#include <util.cuh>

///// Kernels /////

// Vector dot product kernel; VEC - if true =, use vectorized memory access
template <bool VEC>
__global__ void vectorDotProduct_kernel(const float *x, const float *y, float *z, unsigned size)
{
    using VecType = typename std::conditional_t<VEC, float4, float>;
    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(float);

    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned step = blockDim.x * gridDim.x;

    const VecType* x4 = reinterpret_cast<const VecType*>(x);
    const VecType* y4 = reinterpret_cast<const VecType*>(y);
    float sum = 0.0f;

    const unsigned size4 = size / VEC_SIZE;
    for (unsigned i = idx; i < size4; i+= step)
        sum += dot(x4[i], y4[i]);

    sum = blockReduceSum(sum);

    if(idx == 0)
    {
        for(unsigned i = size4 * VEC_SIZE; i < size; ++i)
            sum += x[i] * y[i];
    }

    if(threadIdx.x == 0)
        atomicAdd(z, sum);
}

///// Launch functions /////

template <unsigned blockDim, bool VEC = true>
void launch_vectorDotProduct(const float *x, const float *y, float *z, unsigned size, unsigned maxGridDim)
{
    using VecType = typename std::conditional_t<VEC, float4, float>;
    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(float);

    const unsigned gridDim = std::min(CEIL_DIV(size, VEC_SIZE * blockDim), maxGridDim);
    vectorDotProduct_kernel<VEC><<<gridDim, blockDim>>>(x, y, z, size);
}

#endif // VECTOR_DOT_PRODUCT_KERNELS
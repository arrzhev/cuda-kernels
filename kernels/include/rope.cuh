#ifndef ROPE_KERNELS
#define ROPE_KERNELS

#include <util.cuh>

///// Kernels /////

__global__ void rope_kernel(const float2 *X, float2 *Y, unsigned batchSize, unsigned seqLen, unsigned dim2);

///// Launch functions /////

template <unsigned blockDim>
void launch_rope(const float *X, float *Y, unsigned batchSize, unsigned seqLen, unsigned dim)
{
    const unsigned dim2 = dim / 2U;
    const unsigned gridDim = CEIL_DIV(batchSize * seqLen * dim2, blockDim);
    rope_kernel<<<gridDim, blockDim>>>(reinterpret_cast<const float2*>(X), reinterpret_cast<float2*>(Y), batchSize, seqLen, dim2);
}

#endif // ROPE_KERNELS
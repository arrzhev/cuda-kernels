#ifndef VECTOR_DOT_PRODUCT_KERNELS
#define VECTOR_DOT_PRODUCT_KERNELS

#include <util.cuh>

__global__ void vectorDotProduct_kernel(const float *x, const float *y, float *z, unsigned size);
__global__ void vectorDotProduct4_kernel(const float *x, const float *y, float *z, unsigned size);

///// Launch functions /////

template <unsigned blockDim>
void launch_vectorDotProduct(const float *x, const float *y, float *z, unsigned size, unsigned maxGridDim)
{
    const unsigned gridDim = std::min(CEIL_DIV(size, 4U * blockDim), maxGridDim);
    vectorDotProduct4_kernel<<<gridDim, blockDim>>>(x, y, z, size);
}

#endif // VECTOR_DOT_PRODUCT_KERNELS
#ifndef VECTOR_DOT_PRODUCT_KERNELS
#define VECTOR_DOT_PRODUCT_KERNELS

__global__ void vectorDotProduct_kernel(const float *x, const float *y, float *z, unsigned size);
__global__ void vectorDotProduct4_kernel(const float *x, const float *y, float *z, unsigned size);

#endif // VECTOR_DOT_PRODUCT_KERNELS
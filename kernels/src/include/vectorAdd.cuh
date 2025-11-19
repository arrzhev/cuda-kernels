#ifndef VECTOR_ADD_KERNELS
#define VECTOR_ADD_KERNELS

__global__ void vectorAdd_kernel(const float *x, const float *y, float *z, unsigned size);

#endif // VECTOR_ADD_KERNELS
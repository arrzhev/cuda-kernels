#ifndef MATRIX_VECTOR_MUL_KERNELS
#define MATRIX_VECTOR_MUL_KERNELS

__global__ void matrixVectorMul_naive_kernel(const float* X, const float* y, float* z, unsigned rows, unsigned cols);
__global__ void matrixVectorMul_shared_kernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols);
__global__ void matrixVectorMul_shared4_kernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols);
__global__ void matrixVectorMul_warp_kernel(const float * X, const float *y, float *z, unsigned rows, unsigned cols);
__global__ void matrixVectorMul_warp4_kernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols);

#endif // MATRIX_VECTOR_MUL_KERNELS
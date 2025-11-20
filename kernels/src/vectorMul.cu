#include <vectorMul.cuh>

__global__ void vectorMul_kernel(const float *x, const float *y, float *z, unsigned size)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        z[i] = x[i] * y[i];
}
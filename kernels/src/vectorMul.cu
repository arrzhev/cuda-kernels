#include <vectorMul.cuh>

__global__ void vectorMul_kernel(const float *x, const float *y, float *z, unsigned size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        z[i] = x[i] * y[i];
}
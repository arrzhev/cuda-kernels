#include <vectorAdd.cuh>

__global__ void vectorAdd_kernel(const float *x, const float *y, float *z, unsigned size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        z[i] = x[i] + y[i];
}
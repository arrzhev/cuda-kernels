#include <vectorDotProduct.cuh>
#include <util.cuh>

__global__ void vectorDotProduct_kernel(const float *x, const float *y, float *z, unsigned size)
 {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned step = blockDim.x * gridDim.x;
    float sum = 0.0f;

    for (unsigned i = idx; i < size; i+= step)
            sum += x[i] * y[i];

    sum = blockReduceSum(sum);

    if(threadIdx.x == 0)
        atomicAdd(z, sum);
}

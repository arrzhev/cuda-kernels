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

__global__ void vectorDotProduct4_kernel(const float *x, const float *y, float *z, unsigned size)
{
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned step = blockDim.x * gridDim.x;
    float sum = 0.0f;

    const unsigned size4 = size / 4U;
    for (unsigned i = idx; i < size4; i+= step)
    {
        const float4 tmpX = reinterpret_cast<const float4*>(&x[i * 4U])[0];
        const float4 tmpY = reinterpret_cast<const float4*>(&y[i * 4U])[0];
        sum += (tmpX.x * tmpY.x +
                tmpX.y * tmpY.y +
                tmpX.z * tmpY.z +
                tmpX.w * tmpY.w);
    }

    sum = blockReduceSum(sum);

    if(threadIdx.x == 0)
    {
        if(idx == 0)
        {
            for(unsigned i = size4 * 4U; i < size; ++i)
                sum += x[i] * y[i];
        }
        atomicAdd(z, sum);
    }
}
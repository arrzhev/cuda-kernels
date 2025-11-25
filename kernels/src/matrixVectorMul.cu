#include <matrixVectorMul.cuh>
#include <util.cuh>

__global__ void matrixVectorMul_naive_kernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
 {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        float sum = 0.0f;

        for (unsigned i = 0; i < cols; ++i)
            sum += X[row * cols + i] * y[i];

        z[row] = sum;
    }
}

__global__ void matrixVectorMul_shared_kernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
 {
    unsigned row = blockIdx.x;
    unsigned col = threadIdx.x;
    extern __shared__ float smem[];

    float sum = 0.0f;
    
    for (unsigned i = col; i < cols; i+= blockDim.x)
        sum += X[row * cols + i] * y[i];

    smem[col] = sum;
    __syncthreads();

    for (unsigned stride = blockDim.x>>1; stride > 0; stride>>=1)
    {
        if (col < stride)
            smem[col] += smem[col + stride];
        __syncthreads();
    }

    if(col == 0)
        z[row] = smem[0];
}

__global__ void matrixVectorMul_warp_kernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
{
    unsigned row = blockIdx.x;
    unsigned col = threadIdx.x;

    float sum = 0.0f;

    for (unsigned i = col; i < cols; i+= blockDim.x)
        sum += X[row * cols + i] * y[i];

    sum = blockReduceSum(sum);

    if (col == 0)
        z[row] = sum;
}
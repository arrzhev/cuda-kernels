#include <matrixMul.cuh>
#include <util.cuh>

__global__ void matrixVectorMul_naive_kernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
 {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        double sum = 0.0f;

        for (int i = 0; i < cols; ++i)
            sum += X[row * cols + i] * y[i];

        z[row] = static_cast<float>(sum);
    }
}

__global__ void matrixVectorMul_shared_kernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
 {
    unsigned row = blockIdx.x;
    int col = threadIdx.x;
    extern __shared__ double smem[];

    if (row < rows)
    {
        double sum = 0.0f;

        for (int i = col; i < cols; i+= blockDim.x)
            sum += X[row * cols + i] * y[i];

        smem[col] = sum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            if (col < stride)
                smem[col] = smem[col] + smem[col + stride];
            __syncthreads();
        }

        if(col == 0)
            z[row] = static_cast<float>(smem[0]);
    }
}

__global__ void matrixVectorMul_warp_kernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
{
    unsigned row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows)
    {
        double sum = 0.0f;

        for (int i = col; i < cols; i+= blockDim.x)
            sum += X[row * cols + i] * y[i];

        sum = blockReduceSum(sum);

        if (col == 0)
            z[row] = static_cast<float>(sum);
    }
}
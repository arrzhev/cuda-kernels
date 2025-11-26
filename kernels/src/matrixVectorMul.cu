#include <matrixVectorMul.cuh>
#include <util.cuh>

__global__ void matrixVectorMul_naive_kernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
 {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        float sum = 0.0f;

        for (unsigned i = 0U; i < cols; ++i)
            sum += X[row * cols + i] * y[i];

        z[row] = sum;
    }
}

__global__ void matrixVectorMul_shared_kernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
 {
    const unsigned row = blockIdx.x;
    const unsigned col = threadIdx.x;
    extern __shared__ float smem[];

    float sum = 0.0f;
    
    for (unsigned i = col; i < cols; i+= blockDim.x)
        sum += X[row * cols + i] * y[i];

    smem[col] = sum;
    __syncthreads();

    for (unsigned stride = blockDim.x>>1U; stride > 0U; stride>>=1U)
    {
        if (col < stride)
            smem[col] += smem[col + stride];
        __syncthreads();
    }

    if(col == 0U)
        z[row] = smem[0U];
}

// cols % 4 == 0
__global__ void matrixVectorMul_shared4_kernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
{
    const unsigned row = blockIdx.x;
    const unsigned col = threadIdx.x;
    extern __shared__ float smem[];

    float sum = 0.0f;

    X += (row * cols);
    const unsigned cols4 = cols / 4U;
    for (unsigned i = col; i < cols4; i+= blockDim.x)
    {
        const float4 tmpX = reinterpret_cast<const float4*>(&X[i * 4U])[0];
        const float4 tmpY = reinterpret_cast<const float4*>(&y[i * 4U])[0];
        sum += (tmpX.x * tmpY.x +
                tmpX.y * tmpY.y +
                tmpX.z * tmpY.z +
                tmpX.w * tmpY.w);
    }

    smem[col] = sum;
    __syncthreads();

    for (unsigned stride = blockDim.x>>1U; stride > 0U; stride>>=1U)
    {
        if (col < stride)
            smem[col] += smem[col + stride];

        __syncthreads();
    }

    if(col == 0U)
        z[row] = smem[0U];
}

__global__ void matrixVectorMul_warp_kernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
{
    const unsigned row = blockIdx.x;
    const unsigned col = threadIdx.x;

    float sum = 0.0f;

    for (unsigned i = col; i < cols; i+= blockDim.x)
        sum += X[row * cols + i] * y[i];

    sum = blockReduceSum(sum);

    if (col == 0U)
        z[row] = sum;
}

// cols % 4 == 0
__global__ void matrixVectorMul_warp4_kernel(const float *X, const float *y, float *z, unsigned rows, unsigned cols)
{
    const unsigned row = blockIdx.x;
    const unsigned col = threadIdx.x;

    float sum = 0.0f;

    X += (row * cols);
    const unsigned cols4 = cols / 4U;
    for (unsigned i = col; i < cols4; i+= blockDim.x)
    {
        const float4 tmpX = reinterpret_cast<const float4*>(&X[i * 4U])[0U];
        const float4 tmpY = reinterpret_cast<const float4*>(&y[i * 4U])[0U];
        sum += (tmpX.x * tmpY.x +
                tmpX.y * tmpY.y +
                tmpX.z * tmpY.z +
                tmpX.w * tmpY.w);
    }

    sum = blockReduceSum(sum);

    if (col == 0U)
        z[row] = sum;
}
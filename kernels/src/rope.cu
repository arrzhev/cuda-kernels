#include <rope.cuh>

__global__ void rope_kernel(const float2 *X, float2 *Y, unsigned batchSize, unsigned seqLen, unsigned dim2)
{
    const unsigned tx = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned size = batchSize * seqLen * dim2;

    if(tx < size)
    {
        const float2 x = X[tx];

        constexpr float base = 10000.0f;
        const unsigned row = (tx / dim2) % seqLen;
        const unsigned pair = tx % dim2;
        const float angle = static_cast<float>(row) / powf(base, static_cast<float>(pair) / static_cast<float>(dim2));
        const float cosA = cosf(angle);
        const float sinA = sinf(angle);

        float2 y;
        y.x = x.x * cosA - x.y * sinA;
        y.y = x.y * cosA + x.x * sinA;
        Y[tx] = y;
    }
}

__global__ void rope_cached_kernel(const float2 *X, const float2 *sinCosVec, float2 *Y, unsigned batchSize, unsigned seqLen, unsigned dim2)
{
    const unsigned tx = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned size = batchSize * seqLen * dim2;

    if(tx < size)
    {
        const float2 x = X[tx];

        const unsigned row = (tx / dim2) % seqLen;
        const unsigned pair = tx % dim2;

        const float2 sincosValue = sinCosVec[row * dim2 + pair];
        const float sinA = sincosValue.x;
        const float cosA = sincosValue.y;

        float2 y;
        y.x = x.x * cosA - x.y * sinA;
        y.y = x.y * cosA + x.x * sinA;
        Y[tx] = y;
    }
}
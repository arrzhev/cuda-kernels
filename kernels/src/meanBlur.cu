#include <meanBlur.cuh>

__global__ void meanBlurGray_kernel(const unsigned char* src, unsigned char* dst,
     unsigned rows, unsigned cols, int kernelRadius)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < cols && row < rows)
    {
        float sum = 0.0f;
        float count = 0.0f;
        for (int i = -kernelRadius; i <= kernelRadius; ++i)
        {
            for (int j = -kernelRadius; j <= kernelRadius; ++j)
            {
                const int neighborRow = row + i;
                const int neighborCol = col + j;

                if (neighborRow >= 0 && neighborRow < rows && 
                    neighborCol >= 0 && neighborCol < cols)
                {
                    sum += src[neighborRow * cols + neighborCol];
                    count++;
                }
            }
        }

        dst[row * cols + col] = static_cast<unsigned char>(sum / count);
    }
}

__global__ void meanBlurColor_kernel(const unsigned char* srcR, const unsigned char* srcG, const unsigned char* srcB,
     unsigned char* dstR, unsigned char* dstG, unsigned char* dstB,
     unsigned rows, unsigned cols, int kernelRadius)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < cols && row < rows)
    {
        float sumR = 0.0f;
        float sumG = 0.0f;
        float sumB = 0.0f;
        float count = 0.0f;
        unsigned index = row * cols + col;

        for (int i = -kernelRadius; i <= kernelRadius; ++i)
        {
            for (int j = -kernelRadius; j <= kernelRadius; ++j)
            {
                const int neighborRow = row + i;
                const int neighborCol = col + j;

                if (neighborRow >= 0 && neighborRow < rows && 
                    neighborCol >= 0 && neighborCol < cols)
                {
                    unsigned neighborIndex = neighborRow * cols + neighborCol;
                    sumR += srcR[neighborIndex];
                    sumG += srcG[neighborIndex];
                    sumB += srcB[neighborIndex];
                    count++;
                }
            }
        }

        dstR[index] = static_cast<unsigned char>(sumR / count);
        dstG[index] = static_cast<unsigned char>(sumG / count);
        dstB[index] = static_cast<unsigned char>(sumB / count);
    }
}
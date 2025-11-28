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
        for (int i = max(0, row - kernelRadius); i <= min(rows-1, row + kernelRadius); ++i)
        {
            for (int j = max(0, col - kernelRadius); j <= min(cols - 1, col + kernelRadius); ++j)
            {
                sum += src[i * cols + j];
                count++;
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

        for (int i = max(0, row - kernelRadius); i <= min(rows-1, row + kernelRadius); ++i)
        {
            for (int j = max(0, col - kernelRadius); j <= min(cols - 1, col + kernelRadius); ++j)
            {
                const unsigned neighborIndex = i * cols + j;
                sumR += srcR[neighborIndex];
                sumG += srcG[neighborIndex];
                sumB += srcB[neighborIndex];
                count++;
            }
        }

        const unsigned index = row * cols + col;
        dstR[index] = static_cast<unsigned char>(sumR / count);
        dstG[index] = static_cast<unsigned char>(sumG / count);
        dstB[index] = static_cast<unsigned char>(sumB / count);
    }
}
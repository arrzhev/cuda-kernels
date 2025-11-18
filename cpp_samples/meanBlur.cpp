#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <climits>
#include <assert.h>

#include <meanBlurC.hpp>

void blurChannel(const std::vector<unsigned char>& inImage, std::vector<unsigned char>& outImage, 
    const unsigned rows, const unsigned cols, const unsigned channel, const unsigned kernelSize)
{
    const int kernelRadius = (kernelSize - 1) / 2;
    const int offset = channel * rows * cols;
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            int idx = i * cols + j;
            float sum = 0.0f;
            float count = 0.0f;

            for (int i1 = -kernelRadius; i1 <= kernelRadius; ++i1)
            {
                for (int j1 = -kernelRadius; j1 <= kernelRadius; ++j1)
                {
                    const int neighborRow = i + i1;
                    const int neighborCol = j + j1;

                    if (neighborRow >= 0 && neighborRow < rows && 
                        neighborCol >= 0 && neighborCol < cols)
                    {
                        unsigned neighborIndex = neighborRow * cols + neighborCol;
                        sum += inImage[offset + neighborIndex];
                        count++;
                    }
                }
            }

            outImage[offset + idx] = static_cast<unsigned char>(sum/count);
        }
    }
}

int main(int argc, char **argv)
{
    constexpr unsigned rows = 123;
    constexpr unsigned cols = 321;
    constexpr unsigned channels = 3;
    constexpr unsigned kernelSize = 5;
    constexpr size_t size = rows * cols;
    constexpr size_t totalSize = channels * size;

    std::vector<unsigned char> image;
    std::vector<unsigned char> blurred(totalSize);
    std::vector<unsigned char> cudaBlurred(totalSize);

    std::generate_n(std::back_inserter(image), totalSize, []() { return rand() % (UCHAR_MAX+1); });

    for(int channel = 0; channel < channels; ++channel)
        blurChannel(image, blurred, rows, cols, channel, kernelSize);

    auto blurFunc = channels == 1 ? meanBlurGray : meanBlurColor;
    blurFunc(image.data(), cudaBlurred.data(), rows, cols, kernelSize);

    assert(blurred == cudaBlurred);

    printf("Mean blur C API test passed with: channels=%d kernel=%d size=%dx%d\n", channels, kernelSize, rows, cols);

    return 0;
}
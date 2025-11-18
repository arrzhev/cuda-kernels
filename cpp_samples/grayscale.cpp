#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <climits>
#include <assert.h>

#include <grayscaleC.hpp>
#include "common.hpp"

inline unsigned char rgb2gray(unsigned char r, unsigned char g, unsigned char b)
{
    return static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
}

int main(int argc, char **argv)
{
    constexpr static size_t size = 4096;
    std::vector<unsigned char> image;
    std::vector<unsigned char> grayscale(size);
    std::vector<unsigned char> cudaGrayscale(size);

    std::generate_n(std::back_inserter(image), 3*size, []() { return rand() % (UCHAR_MAX+1); });

    for(int i = 0, i3 = 0; i < size; i++, i3+=3)
        grayscale[i] = rgb2gray(image[i3], image[i3+1], image[i3 + 2]);

    rgb2grayInterleaved(image.data(), cudaGrayscale.data(), size);

    assert(compareVectors(grayscale, cudaGrayscale, 1.0f));

    std::cout << "Grayscale interleaved C api test passed!" << std::endl;

    for(auto r = image.begin(), g = r + size, b = g + size; auto &out : grayscale)
        out = rgb2gray(*r++, *g++, *b++);

    rgb2grayPlanar(image.data(), cudaGrayscale.data(), size);

    assert(compareVectors(grayscale, cudaGrayscale, 1.0f));

    std::cout << "Grayscale planar C api test passed!" << std::endl;

    return 0;
}
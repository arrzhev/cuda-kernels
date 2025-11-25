#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <assert.h>

#include <matrixVectorMulC.hpp>
#include "common.hpp"

int main(int argc, char **argv)
{
    constexpr size_t rows = 1234;
    constexpr size_t cols = 4321;
    std::vector<float> x(rows * cols);
    std::vector<float> y(cols);
    std::vector<float> result(rows);
    std::vector<float> cudaResult(rows);

    std::generate(x.begin(), x.end(), [](){return rand() / (float)RAND_MAX;});
    std::generate(y.begin(), y.end(), [](){return rand() / (float)RAND_MAX;});

    for(unsigned row = 0; row < rows; ++row)
    {
        float sum = 0.0;
        for(unsigned col = 0; col < cols; ++col)
            sum += x[row * cols + col] * y[col];
        result[row] = sum;
    }

    matrixVectorMul(x.data(), y.data(), cudaResult.data(), rows, cols);

    assert(compareVectors(result, cudaResult));

    std::cout << "Matrix x Vector multiplication C api test passed!" << std::endl;

    return 0;
}
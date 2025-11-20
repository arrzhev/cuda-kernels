#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <assert.h>

#include <vectorDotProductC.hpp>
#include "common.hpp"

int main(int argc, char **argv)
{
    constexpr size_t size = 12345;
    std::vector<float> x(size);
    std::vector<float> y(size);
    float result;
    float cudaResult;

    std::generate(x.begin(), x.end(), [](){return rand() / (float)RAND_MAX;});
    std::generate(y.begin(), y.end(), [](){return rand() / (float)RAND_MAX;});

    for(unsigned i = 0U; i < size; ++i)
        result += x[i] * y[i];

    vectorDotProduct(x.data(), y.data(), &cudaResult, size);

    assert(std::fabs(result - cudaResult) < 1e-3);

    std::cout << "Vector dot product C api test passed!" << std::endl;

    return 0;
}
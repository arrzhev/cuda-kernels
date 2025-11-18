#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <assert.h>

#include <vectorAddC.hpp>
#include "common.hpp"

int main(int argc, char **argv)
{
    constexpr size_t size = 4096;
    std::vector<float> x(size);
    std::vector<float> y(size);
    std::vector<float> result(size);
    std::vector<float> cudaResult(size);

    std::generate(x.begin(), x.end(), [](){return rand() / (float)RAND_MAX;});
    std::generate(y.begin(), y.end(), [](){return rand() / (float)RAND_MAX;});

    std::transform(x.begin(), x.end(), y.begin(), result.begin(), std::plus<float>());

    vectorAdd(x.data(), y.data(), cudaResult.data(), size);

    assert(compareVectors(result, cudaResult));

    std::cout << "Vector add C api test passed!" << std::endl;

    return 0;
}
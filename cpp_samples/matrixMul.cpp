#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <assert.h>

#include <matrixMulC.hpp>
#include "common.hpp"

int main(int argc, char **argv)
{
    constexpr size_t M = 1234;
    constexpr size_t N = 4321;
    constexpr size_t K = 1111;
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> result(M * N);
    std::vector<float> cudaResult(M * N);

    std::generate(A.begin(), A.end(), [](){return rand() / (float)RAND_MAX;});
    std::generate(B.begin(), B.end(), [](){return rand() / (float)RAND_MAX;});

    for(unsigned row = 0; row < M; ++row)
    {
        for(unsigned col = 0; col < N; ++col)
        {
            float sum = 0.0f;
            for (unsigned i = 0; i < K; ++i)
                sum += A[row * K + i] * B[i * N + col];
            result[row * N + col] = sum;
        }
    }

    matrixMul(A.data(), B.data(), cudaResult.data(), M, N, K);

    assert(compareVectors(result, cudaResult));

    std::cout << "Matrix x Matrix multiplication C api test passed!" << std::endl;

    return 0;
}
#ifndef MATRIX_MUL_TORCH
#define MATRIX_MUL_TORCH

#include <torch/torch.h>

torch::Tensor matrixMul(torch::Tensor x, torch::Tensor y);
torch::Tensor matrixMulNaive(torch::Tensor x, torch::Tensor y);
torch::Tensor matrixMulCoalescing(torch::Tensor x, torch::Tensor y);
torch::Tensor matrixMulTiled(torch::Tensor x, torch::Tensor y);
torch::Tensor matrixMulTiled1D(torch::Tensor x, torch::Tensor y);
torch::Tensor matrixMulTiled2D(torch::Tensor x, torch::Tensor y);
torch::Tensor matrixMulTiled2D4(torch::Tensor x, torch::Tensor y);

#endif // MATRIX_MUL_TORCH
#ifndef MATRIX_VECTOR_MUL_TORCH
#define MATRIX_VECTOR_MUL_TORCH

#include <torch/torch.h>

torch::Tensor matrixVectorMul(torch::Tensor x, torch::Tensor y);
torch::Tensor matrixVectorMulNaive(torch::Tensor x, torch::Tensor y);
torch::Tensor matrixVectorMulShared(torch::Tensor x, torch::Tensor y);
torch::Tensor matrixVectorMulShared4(torch::Tensor x, torch::Tensor y);
torch::Tensor matrixVectorMulWarp(torch::Tensor x, torch::Tensor y);
torch::Tensor matrixVectorMulWarp4(torch::Tensor x, torch::Tensor y);

#endif // MATRIX_VECTOR_MUL_TORCH
#ifndef MATMUL_NN_TORCH
#define MATMUL_NN_TORCH

#include <torch/torch.h>

std::vector<torch::Tensor> matrixMulBias(torch::Tensor A, torch::Tensor B, torch::Tensor bias, bool useReLU, bool transC);
torch::Tensor matrixMulReLU(torch::Tensor A, torch::Tensor AR, torch::Tensor B, bool transC);

#endif // MATMUL_NN_TORCH
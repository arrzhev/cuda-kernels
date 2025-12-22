#ifndef MATRIX_REDUCTION_TORCH
#define MATRIX_REDUCTION_TORCH

#include <torch/torch.h>

torch::Tensor matrixRowReduction(torch::Tensor A);
torch::Tensor matrixRowReductionReLU(torch::Tensor A, torch::Tensor AR);

#endif // MATRIX_REDUCTION_TORCH
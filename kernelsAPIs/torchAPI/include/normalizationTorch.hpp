#ifndef NORMALIZATION_TORCH
#define NORMALIZATION_TORCH

#include <torch/torch.h>

torch::Tensor layerNorm(torch::Tensor X, torch::Tensor W, torch::Tensor B, float eps = 1e-5);
torch::Tensor RMSNorm(torch::Tensor X, torch::Tensor W, float eps = 1e-5);

#endif // NORMALIZATION_TORCH
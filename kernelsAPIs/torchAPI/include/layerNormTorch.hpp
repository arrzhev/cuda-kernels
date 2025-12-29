#ifndef LAYER_NORM_TORCH
#define LAYER_NORM_TORCH

#include <torch/torch.h>

torch::Tensor layerNorm(torch::Tensor X, torch::Tensor W, torch::Tensor B, float eps = 1e-5);

#endif // LAYER_NORM_TORCH
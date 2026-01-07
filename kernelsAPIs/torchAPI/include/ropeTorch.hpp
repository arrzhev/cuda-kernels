#ifndef ROPE_TORCH
#define ROPE_TORCH

#include <torch/torch.h>

torch::Tensor rope(torch::Tensor X);
torch::Tensor rope_cached(torch::Tensor X, torch::Tensor sinCosVec);

#endif // ROPE_TORCH
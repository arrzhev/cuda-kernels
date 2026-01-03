#ifndef ROPE_TORCH
#define ROPE_TORCH

#include <torch/torch.h>

torch::Tensor rope(torch::Tensor X);

#endif // ROPE_TORCH
#ifndef TENSOR_ADD_TORCH
#define TENSOR_ADD_TORCH

#include <torch/torch.h>

torch::Tensor tensorAdd(torch::Tensor x, torch::Tensor y);

#endif // TENSOR_ADD_TORCH
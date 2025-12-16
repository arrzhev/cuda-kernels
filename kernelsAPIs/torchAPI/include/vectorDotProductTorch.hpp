#ifndef VECTOR_DOT_PRODUCT_TORCH
#define VECTOR_DOT_PRODUCT_TORCH

#include <torch/torch.h>

torch::Tensor vectorDotProduct(torch::Tensor x, torch::Tensor y);

#endif // VECTOR_DOT_PRODUCT_TORCH
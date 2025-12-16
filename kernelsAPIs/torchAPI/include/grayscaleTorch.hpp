#ifndef GRAYSCALE_TORCH
#define GRAYSCALE_TORCH

#include <torch/torch.h>

torch::Tensor rgb2gray(torch::Tensor rgbImage);

#endif // GRAYSCALE_TORCH
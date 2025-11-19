#ifndef MEAN_BLUR_TORCH
#define MEAN_BLUR_TORCH

#include <torch/torch.h>

torch::Tensor meanBlur(torch::Tensor image, const unsigned kernelSize);

#endif // MEAN_BLUR_TORCH
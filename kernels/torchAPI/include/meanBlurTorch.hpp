#include <torch/torch.h>

torch::Tensor meanBlur(torch::Tensor image, const unsigned kernelSize);
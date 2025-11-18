#include <torch/torch.h>

torch::Tensor tensorMulNaive(torch::Tensor x, torch::Tensor y);
torch::Tensor tensorMulShared(torch::Tensor x, torch::Tensor y);
torch::Tensor tensorMulWarp(torch::Tensor x, torch::Tensor y);
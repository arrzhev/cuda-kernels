#ifndef MATMUL_TORCH
#define MATMUL_TORCH

#include <torch/torch.h>

torch::Tensor matrixMul(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulNaive(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulNaiveK(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulCoalescing(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulCoalescingK(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulBTiles(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulBTilesK(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulBTilesDBuf(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulTTiles1D(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulTTiles1DK(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulTTiles1DDBuf(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulTTiles2D(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulTTiles2DK(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulTTiles2DDBuf(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulTTiles2DVec(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulTTiles2DVecK(torch::Tensor x, torch::Tensor y);
torch::Tensor matmulTTiles2DDBufVec(torch::Tensor x, torch::Tensor y);

#endif // MATMUL_TORCH
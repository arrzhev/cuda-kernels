#ifndef MATMUL_TORCH
#define MATMUL_TORCH

#include <torch/torch.h>

torch::Tensor matrixMul(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulNaive(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulNaiveK(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulCoalescing(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulCoalescingK(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulBTiles(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulBTilesK(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulBTilesDBuf(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulBTilesDBufK(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulTTiles1D(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulTTiles1DK(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulTTiles1DDBuf(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulTTiles1DDBufK(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulTTiles2D(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulTTiles2DK(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulTTiles2DDBuf(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulTTiles2DDBufK(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulTTiles2DVec(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulTTiles2DVecK(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulTTiles2DDBufVec(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulTTiles2DDBufVecK(torch::Tensor A, torch::Tensor B, bool transC = false);
torch::Tensor matmulBTilesVecWMMA(torch::Tensor A, torch::Tensor B, bool transC = false);

#endif // MATMUL_TORCH
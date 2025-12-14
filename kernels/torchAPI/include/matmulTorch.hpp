#ifndef MATMUL_TORCH
#define MATMUL_TORCH

#include <torch/torch.h>

torch::Tensor matrixMul(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulNaive(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulNaiveK(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulCoalescing(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulCoalescingK(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulBTiles(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulBTilesK(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulBTilesDBuf(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulBTilesDBufK(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulTTiles1D(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulTTiles1DK(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulTTiles1DDBuf(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulTTiles1DDBufK(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulTTiles2D(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulTTiles2DK(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulTTiles2DDBuf(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulTTiles2DDBufK(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulTTiles2DVec(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulTTiles2DVecK(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulTTiles2DDBufVec(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulTTiles2DDBufVecK(torch::Tensor A, torch::Tensor B);
torch::Tensor matmulBTilesVecWMMA(torch::Tensor A, torch::Tensor B);

#endif // MATMUL_TORCH
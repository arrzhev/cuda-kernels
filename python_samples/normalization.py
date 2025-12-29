import torch
import torch.nn.functional as F

import torch_extension

if __name__ == '__main__':
    M = 1234
    D = 4321
    X = torch.randn(M, D, dtype=torch.float32, device='cuda')
    W = torch.randn(D, dtype=torch.float32, device='cuda')
    B = torch.randn(D, dtype=torch.float32, device='cuda')

    y_torch = F.layer_norm(X, (D,), W, B)
    y_extension = torch_extension.layer_norm(X, W, B)

    print(y_torch)
    print(y_extension)
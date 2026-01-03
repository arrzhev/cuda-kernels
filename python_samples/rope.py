import torch
import torch.nn.functional as F

import torch_extension

if __name__ == '__main__':
    B = 123
    S = 123
    D = 123
    X = torch.randn(B, S, D, dtype=torch.float32, device='cuda')

    y_extension = torch_extension.rope(X).cpu()

    print(y_extension)
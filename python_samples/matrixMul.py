import torch

import torch_extension

if __name__ == '__main__':
    dtype = torch.float16
    M = 1234
    N = 4321
    K = 1111
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    y = torch.randn(K, N, dtype=dtype, device="cuda")

    z_torch = x @ y
    z_extension = torch_extension.matmul(x, y)

    print('Matmul test')
    print(f'Torch result - {z_torch}')
    print(f'Extension result - {z_extension}')
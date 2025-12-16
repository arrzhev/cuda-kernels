import torch

import torch_extension

if __name__ == '__main__':
    size = 4096
    x = torch.randn(size, device="cuda")
    y = torch.randn(size, device="cuda")

    z_torch = x @ y
    z_extension = torch_extension.vector_dot_product(x, y)

    print('Vector x Vector multiplication')
    print(f'Torch result - {z_torch}')
    print(f'Extension result - {z_extension}')
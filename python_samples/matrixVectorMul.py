import torch

import torch_extension

if __name__ == '__main__':
    size1 = 1234
    size2 = 4321
    x = torch.randn(size1, size2, device="cuda")
    y = torch.randn(size2, device="cuda")

    z_torch = x @ y
    z_extension = torch_extension.matrix_vector_mul(x, y)

    print('Matrix x Vector multiplication')
    print(f'Torch result - {z_torch}')
    print(f'Extension result - {z_extension}')
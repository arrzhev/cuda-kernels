import pytest
import torch
import torch.utils.benchmark as benchmark

import torch_extension

@pytest.mark.unit
@pytest.mark.parametrize("size", [1, 10, 256, 1213, 4096, 8000, 12345])
def test_vector_dot_product(size):
    x = torch.randn(size, device="cuda")
    y = torch.randn(size, device="cuda")

    z_torch = x @ y
    z_extension = torch_extension.vector_dot_product(x, y)

    torch.testing.assert_close(z_torch, z_extension, atol=1e-3, rtol=1e-3)

@pytest.mark.performance
def test_perf_vector_dot_product():
    results = []
    sizes = [1, 10, 256, 1213, 4096, 8000, 8000000]

    for size in sizes:
        label = 'Vector dot product'
        sub_label = f'Size: {size}'
        x = torch.randn(size, device="cuda")
        y = torch.randn(size, device="cuda")
        results.append(benchmark.Timer(
            stmt='x @ y',
            setup='',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='torch',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.vector_dot_product(x, y)',
            setup='import torch_extension',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='extension',
        ).blocked_autorange())

    compare = benchmark.Compare(results)
    compare.print()

if __name__ == '__main__':
    size = 4096
    x = torch.randn(size, device="cuda")
    y = torch.randn(size, device="cuda")

    z_torch = x @ y
    z_extension = torch_extension.vector_dot_product(x, y)

    print('Vector x Vector multiplication')
    print(f'Torch result - {z_torch}')
    print(f'Extension result - {z_extension}')
import pytest
import torch
import torch.utils.benchmark as benchmark

import torch_extension

@pytest.mark.unit
@pytest.mark.parametrize("size", [1, 10, 64, 123])
@pytest.mark.parametrize("dim", [1, 3])
def test_tensor_add(size, dim):
    x = torch.randn([size for i in range(dim)], device="cuda")
    y = torch.randn([size for i in range(dim)], device="cuda")

    z_torch = x + y
    z_extension = torch_extension.tensor_add(x, y)

    torch.testing.assert_close(z_torch, z_extension, atol=1e-5, rtol=1e-5)

@pytest.mark.performance
def test_perf_tensor_add():
    results = []
    sizes = [1, 10, 256, 1213, 4096, 8000, 8000000]

    for size in sizes:
        label = 'Tensor Add'
        sub_label = f'[{size}]'
        x = torch.randn(size, device="cuda")
        y = torch.randn(size, device="cuda")
        results.append(benchmark.Timer(
            stmt='x + y',
            setup='',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='torch',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.tensor_add(x, y)',
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

    z_torch = x + y
    z_extension = torch_extension.tensor_add(x, y)

    print(f"Torch result - {z_torch}")
    print(f"Extension result - {z_extension}")
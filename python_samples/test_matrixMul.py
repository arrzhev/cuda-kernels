import pytest
import torch
import torch.utils.benchmark as benchmark
import itertools

import torch_extension

@pytest.mark.unit
@pytest.mark.parametrize("M, N, K", [(1, 1, 1), (1, 1, 1234), (1, 1234, 1), (1234, 1, 1), (10, 10, 10), (64, 64, 64), (256,256, 256)])
def test_matrix_mul(M, N, K):
    x = torch.randn(M, K, device="cuda")
    y = torch.randn(K, N, device="cuda")

    z_torch = x @ y
    z_extension = torch_extension.matrix_mul(x, y)
    z_extension_naive = torch_extension.matrix_mul_naive(x, y)
    z_extension_coalescing = torch_extension.matrix_mul_coalescing(x, y)
    z_extension_tiled = torch_extension.matrix_mul_tiled(x, y)
    z_extension_tiled_1D = torch_extension.matrix_mul_tiled_1D(x, y)
    z_extension_tiled_2D = torch_extension.matrix_mul_tiled_2D(x, y)

    torch.testing.assert_close(z_torch, z_extension, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(z_torch, z_extension_naive, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(z_torch, z_extension_coalescing, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(z_torch, z_extension_tiled, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(z_torch, z_extension_tiled_1D, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(z_torch, z_extension_tiled_2D, atol=1e-3, rtol=1e-3)

@pytest.mark.performance
def test_perf_matrix_mul():
    results = []

    for M, N, K in [(1, 1, 1), (1, 1, 1234), (1, 1234, 1), (1234, 1, 1), (10, 10, 10), (64, 64, 64), (256,256, 256), (512, 512, 512)]:
        label = 'Matrix Mul'
        sub_label = f'Matrix1: {M}x{K}; Matrix2: {K}x{N}'
        x = torch.randn(M, K, device="cuda")
        y = torch.randn(K, N, device="cuda")
        results.append(benchmark.Timer(
            stmt='x @ y',
            setup='',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='torch',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.matrix_mul_naive(x, y)',
            setup='import torch_extension',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='ext naive',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.matrix_mul_coalescing(x, y)',
            setup='import torch_extension',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='ext coalescing',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.matrix_mul_tiled(x, y)',
            setup='import torch_extension',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='ext tiled',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.matrix_mul_tiled_1D(x, y)',
            setup='import torch_extension',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='ext tiled 1D',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.matrix_mul_tiled_2D(x, y)',
            setup='import torch_extension',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='ext tiled 2D',
        ).blocked_autorange())

        # results.append(benchmark.Timer(
        #     stmt='torch_extension.matrix_mul(x, y)',
        #     setup='import torch_extension',
        #     globals={'x': x, 'y': y},
        #     label=label,
        #     sub_label=sub_label,
        #     description='ext opt',
        # ).blocked_autorange())

    compare = benchmark.Compare(results)
    compare.print()

if __name__ == '__main__':
    M = 1234
    N = 4321
    K = 1111
    x = torch.randn(M, K, device="cuda")
    y = torch.randn(K, N, device="cuda")

    z_torch = x @ y
    z_extension = torch_extension.matrix_mul_tiled_2D(x, y)

    print('Matmul test')
    print(f'Torch result - {z_torch}')
    print(f'Extension result - {z_extension}')
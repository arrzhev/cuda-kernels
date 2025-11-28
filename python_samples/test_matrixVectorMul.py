import pytest
import torch
import torch.utils.benchmark as benchmark

import torch_extension

@pytest.mark.unit
@pytest.mark.parametrize("size1, size2", [(1, 1), (1, 10), (10, 1), (1, 1111), (1111, 1),
                                          (10, 10), (10, 1111), (1111, 10), (1111, 1111),
                                          (4096, 4096), (8000, 8000), (8001, 8001),
                                         ])
def test_matrix_vector_mul(size1, size2):
    x = torch.randn(size1, size2, device="cuda")
    y = torch.randn(size2, device="cuda")

    z_torch = x @ y
    z_extension = torch_extension.matrix_vector_mul(x, y)
    z_extension_naive = torch_extension.matrix_vector_mul_naive(x, y)
    z_extension_shared = torch_extension.matrix_vector_mul_shared(x, y)
    z_extension_warp = torch_extension.matrix_vector_mul_warp(x, y)

    torch.testing.assert_close(z_torch, z_extension, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(z_torch, z_extension_naive, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(z_torch, z_extension_shared, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(z_torch, z_extension_warp, atol=1e-3, rtol=1e-3)

@pytest.mark.performance
def test_perf_matrix_vector_mul():
    results = []
    sizes = [(1, 1), (1, 10), (10, 1), (1, 1111), (1111, 1),
             (10, 10), (10, 1111), (1111, 10), (1111, 1111),
             (4096, 4096), (8000, 8000), (8001, 8001),
            ]

    for size1, size2 in sizes:
        label = 'Matrix x Vector'
        sub_label = f'Matrix: {size1}x{size2}; Vector: {size2}'
        x = torch.randn(size1, size2, device="cuda")
        y = torch.randn(size2, device="cuda")
        results.append(benchmark.Timer(
            stmt='x @ y',
            setup='',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='torch',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.matrix_vector_mul_naive(x, y)',
            setup='import torch_extension',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='ext naive',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.matrix_vector_mul_shared(x, y)',
            setup='import torch_extension',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='ext shared',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.matrix_vector_mul_warp(x, y)',
            setup='import torch_extension',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='ext warp',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.matrix_vector_mul(x, y)',
            setup='import torch_extension',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='ext opt',
        ).blocked_autorange())

    compare = benchmark.Compare(results)
    compare.print()

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
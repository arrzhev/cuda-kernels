import pytest
import torch
import torch.utils.benchmark as benchmark

import torch_extension

FUNCS = [
         'matrix_vector_mul',
         'matrix_vector_mul_naive',
         'matrix_vector_mul_shared',
         'matrix_vector_mul_warp',
        ]

SIZES = [
         (1, 1), (1, 10), (10, 1), (1, 1111), (1111, 1),
         (10, 10), (10, 1111), (1111, 10), (1111, 1111),
         (4096, 4096), (8000, 8000), (8001, 8001),
        ]

@pytest.mark.unit
@pytest.mark.parametrize("func", FUNCS)
@pytest.mark.parametrize("size1, size2", SIZES)
def test_matrix_vector_mul(func, size1, size2):
    x = torch.randn(size1, size2, device="cuda")
    y = torch.randn(size2, device="cuda")

    z_torch = x @ y
    z_extension = getattr(torch_extension, func)(x, y)

    torch.testing.assert_close(z_torch, z_extension, atol=1e-3, rtol=1e-3)

@pytest.mark.performance
def test_perf_matrix_vector_mul():
    results = []

    for size1, size2 in SIZES:
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

        for func in FUNCS:
            results.append(benchmark.Timer(
            stmt='torch_extension.' + func + '(x, y)',
            setup='',
            globals={'torch_extension': torch_extension, 'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description = func,
            ).blocked_autorange())

    compare = benchmark.Compare(results)
    compare.print()
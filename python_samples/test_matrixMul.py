import pytest
import torch
import torch.utils.benchmark as benchmark

import torch_extension

def gen_mats(M, N, K, layout='tt', dtype=torch.float32, device='cuda'):
    if layout[0] == 't':
        x = torch.randn(M, K, dtype=dtype, device=device)
    else:
        x = torch.randn(K, M, dtype=dtype, device=device).T

    if layout[1] == 't':
        y = torch.randn(K, N, dtype=dtype, device=device)
    else:
        y = torch.randn(N, K, dtype=dtype, device=device).T

    return x, y

LAYOUTS = ['tt', 'tn', 'nt', 'nn']

ALL_FUNCS = [
            'matmul',
            'matmul_naive', 'matmul_coalescing',
            'matmul_naive_K', 'matmul_coalescing_K',
            'matmul_BTiles', 'matmul_BTiles_DBuf',
            'matmul_BTiles_K', 'matmul_BTiles_DBuf_K',
            'matmul_TTiles_1D', 'matmul_TTiles_1D_DBuf',
            'matmul_TTiles_1D_K', 'matmul_TTiles_1D_DBuf_K',
            'matmul_TTiles_2D', 'matmul_TTiles_2D_DBuf',
            'matmul_TTiles_2D_K', 'matmul_TTiles_2D_DBuf_K',
            'matmul_TTiles_2D_vec', 'matmul_TTiles_2D_DBuf_vec',
            'matmul_TTiles_2D_vec_K', 'matmul_TTiles_2D_DBuf_vec_K',
            ]

RT_FUNCS = [
             'matmul',
             'matmul_naive', 'matmul_naive_K',
             'matmul_BTiles', 'matmul_BTiles_K',
             'matmul_TTiles_1D', 'matmul_TTiles_2D',
             'matmul_TTiles_2D_vec', 'matmul_TTiles_2D_DBuf_vec',
             'matmul_TTiles_2D_vec_K', 'matmul_TTiles_2D_DBuf_vec_K',
           ]

IRREGULAR_SIZES = [(1, 1, 1), (1, 1, 1234), (1, 1234, 1), (1234, 1, 1), (10, 10, 10), (85, 77, 43), (123, 123, 123), (513, 512, 511)]
SCALE_SIZES = [(64, 64, 64), (256,256, 256), (512, 512, 512)]
SCALE_K_SIZES = [(128, 128, 256), (128, 128, 512), (128, 128, 1024), (128, 128, 2048)]

@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize("M, N, K", IRREGULAR_SIZES + SCALE_SIZES + SCALE_K_SIZES)
def test_matmul(dtype, layout, func, M, N, K):
    x, y = gen_mats(M, N, K, layout, dtype)

    z_torch = x @ y
    z_extension = getattr(torch_extension, func)(x, y)

    torch.testing.assert_close(z_torch, z_extension, atol=1e-3, rtol=1e-3)

@pytest.mark.performance
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_perf_matmul(dtype, layout):
    results = []

    for M, N, K in IRREGULAR_SIZES + SCALE_SIZES + SCALE_K_SIZES:
        label = 'Matrix Mul ' + layout
        sub_label = f'Matrix1: {M}x{K}; Matrix2: {K}x{N}'
        x, y = gen_mats(M, N, K, layout, dtype)

        results.append(benchmark.Timer(
            stmt='x @ y',
            setup='',
            globals={'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description='torch',
        ).blocked_autorange())

        for func in RT_FUNCS:
            results.append(benchmark.Timer(
            stmt='torch_extension.' + func + '(x, y)',
            setup='',
            globals={'torch_extension': torch_extension, 'x': x, 'y': y},
            label=label,
            sub_label=sub_label,
            description = func,
            ).blocked_autorange())

    compare = benchmark.Compare(results)
    print('\n')
    compare.print()

if __name__ == '__main__':
    dtype = torch.float16
    M = 1234
    N = 4321
    K = 1111
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    y = torch.randn(K, N, dtype=dtype, device="cuda")

    z_torch = x @ y
    z_extension = torch_extension.matmul_naive(x, y)
    torch.testing.assert_close(z_torch, z_extension, atol=1e-6, rtol=1e-6)
    print('Matmul test')
    print(f'Torch result - {z_torch}')
    print(f'Extension result - {z_extension}')
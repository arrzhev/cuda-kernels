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

FUNCS = [
         'matmul',
         'matmul_naive', 'matmul_coalescing',
         'matmul_BTiles', 'matmul_BTiles_DBuf',
         'matmul_TTiles_1D', 'matmul_TTiles_1D_DBuf',
         'matmul_TTiles_2D', 'matmul_TTiles_2D_DBuf',
         'matmul_TTiles_2D_vec', 'matmul_TTiles_2D_DBuf_vec',
        ]

@pytest.mark.unit
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("func", FUNCS)
@pytest.mark.parametrize("M, N, K", [(1, 1, 1), (1, 1, 1234), (1, 1234, 1), (1234, 1, 1),
                                     (10, 10, 10), (64, 64, 64), (256,256, 256), (257, 257, 257)
                                    ])
def test_matmul(layout, func, M, N, K):
    x, y = gen_mats(M, N, K, layout, torch.float32)

    z_torch = x @ y
    z_extension = getattr(torch_extension, func)(x, y)

    torch.testing.assert_close(z_torch, z_extension, atol=1e-3, rtol=1e-3)

@pytest.mark.performance
@pytest.mark.parametrize("layout", LAYOUTS)
def test_perf_matmul(layout):
    results = []

    for M, N, K in [(1, 1, 1), (1, 1, 1234), (1, 1234, 1), (1234, 1, 1),
                    (10, 10, 10), (64, 64, 64), (256,256, 256), (512, 512, 512), (513, 513, 513)]:
        label = 'Matrix Mul ' + layout
        sub_label = f'Matrix1: {M}x{K}; Matrix2: {K}x{N}'
        x, y = gen_mats(M, N, K, layout, torch.float32)

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
    print('\n')
    compare.print()

if __name__ == '__main__':
    M = 1234
    N = 4321
    K = 1111
    x = torch.randn(M, K, device="cuda")
    y = torch.randn(K, N, device="cuda")

    z_torch = x @ y
    z_extension = torch_extension.matmul(x, y)

    print('Matmul test')
    print(f'Torch result - {z_torch}')
    print(f'Extension result - {z_extension}')
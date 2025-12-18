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
DTYPES = [torch.float32, torch.float16]
TRANS_C = [False, True]

GEN_FUNCS = [
            'matmul',
            # 'matmul_naive', 'matmul_coalescing',
            # 'matmul_naive_K', 'matmul_coalescing_K',
            # 'matmul_BTiles', 'matmul_BTiles_DBuf',
            # 'matmul_BTiles_K', 'matmul_BTiles_DBuf_K',
            # 'matmul_TTiles_1D', 'matmul_TTiles_1D_DBuf',
            # 'matmul_TTiles_1D_K', 'matmul_TTiles_1D_DBuf_K',
            # 'matmul_TTiles_2D', 'matmul_TTiles_2D_DBuf',
            # 'matmul_TTiles_2D_K', 'matmul_TTiles_2D_DBuf_K',
            'matmul_TTiles_2D_vec', 'matmul_TTiles_2D_DBuf_vec',
            # 'matmul_TTiles_2D_vec_K', 'matmul_TTiles_2D_DBuf_vec_K',
            ]

REDUCED_FUNCS = [
                # 'matmul_BTiles_vec_wmma',
                ]

RT_GEN_FUNCS = [
            'matmul',
            # 'matmul_naive', 'matmul_naive_K',
            # 'matmul_BTiles', 'matmul_BTiles_K',
            # 'matmul_TTiles_1D', 'matmul_TTiles_2D',
            'matmul_TTiles_2D_vec', 'matmul_TTiles_2D_DBuf_vec',
            # 'matmul_TTiles_2D_vec_K', 'matmul_TTiles_2D_DBuf_vec_K',
           ]

RT_REDUCED_FUNCS = [
                    # 'matmul_BTiles_vec_wmma',
                   ]

IRREGULAR_SIZES = [(1, 1, 1), (1, 1, 1234), (1, 1234, 1), (1234, 1, 1), (10, 10, 10), (85, 77, 43), (123, 123, 123), (513, 512, 511)]
SCALE_SIZES = [(64, 64, 64), (256,256, 256), (512, 512, 512)]
SCALE_K_SIZES = [(128, 128, 256), (128, 128, 512), (128, 128, 1024), (128, 128, 2048)]
SPECIAL_SIZES = [(32, 256, 784), (32, 128, 256), (32, 10, 128), (64, 256, 784), (64, 128, 256), (64, 10, 128)]

@pytest.mark.unit
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("transC", TRANS_C)
@pytest.mark.parametrize("func", GEN_FUNCS + REDUCED_FUNCS)
@pytest.mark.parametrize("M, N, K", IRREGULAR_SIZES + SCALE_SIZES + SCALE_K_SIZES)
def test_matmul_gen(dtype, layout, transC, func, M, N, K):
    execute_test = not (dtype == torch.float32 and func in REDUCED_FUNCS)
    if execute_test:
        x, y = gen_mats(M, N, K, layout, dtype)

        z_torch = x @ y
        z_extension = getattr(torch_extension, func)(x, y, transC)

        torch.testing.assert_close(z_torch, z_extension, atol=1e-3, rtol=1e-3)

@pytest.mark.unit
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("transC", TRANS_C)
@pytest.mark.parametrize("M, N, K", IRREGULAR_SIZES + SCALE_SIZES + SPECIAL_SIZES)
def test_matmul_bias(dtype, layout, transC, M, N, K):
    x, y = gen_mats(M, N, K, layout, dtype)
    b = torch.randn(N, dtype=dtype, device='cuda')

    z_torch = x @ y + b
    z_extension = torch_extension.matmul_bias(x, y, b, transC)

    torch.testing.assert_close(z_torch, z_extension, atol=1e-2, rtol=1e-2)

@pytest.mark.performance
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("transC", TRANS_C)
def test_perf_matmul_gen(dtype, layout, transC):
    results = []

    for M, N, K in IRREGULAR_SIZES + SCALE_SIZES + SCALE_K_SIZES:
        label = f'Matrix Mul {str(dtype)} {layout} {"CT" if transC else ""}'
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

        for func in RT_GEN_FUNCS:
            results.append(benchmark.Timer(
            stmt='torch_extension.' + func + '(x, y, transC)',
            setup='',
            globals={'torch_extension': torch_extension, 'x': x, 'y': y, 'transC': transC},
            label=label,
            sub_label=sub_label,
            description = func,
            ).blocked_autorange())
        
        if dtype == torch.float16:
            for func in RT_REDUCED_FUNCS:
                results.append(benchmark.Timer(
                stmt='torch_extension.' + func + '(x, y, transC)',
                setup='',
                globals={'torch_extension': torch_extension, 'x': x, 'y': y, 'transC': transC},
                label=label,
                sub_label=sub_label,
                description = func,
                ).blocked_autorange())

    compare = benchmark.Compare(results)
    print('\n')
    compare.print()

@pytest.mark.performance
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("transC", TRANS_C)
def test_perf_matmul_bias(dtype, layout, transC):
    results = []

    for M, N, K in IRREGULAR_SIZES + SCALE_SIZES + SPECIAL_SIZES:
        label = f'Matrix Mul with Bias {str(dtype)} {layout} {"CT" if transC else ""}'
        sub_label = f'Matrix1: {M}x{K}; Matrix2: {K}x{N}'
        x, y = gen_mats(M, N, K, layout, dtype)
        b = torch.randn(N, dtype=dtype, device='cuda')

        results.append(benchmark.Timer(
            stmt='x @ y + b',
            setup='',
            globals={'x': x, 'y': y, 'b': b},
            label=label,
            sub_label=sub_label,
            description='torch',
        ).blocked_autorange())

        results.append(benchmark.Timer(
        stmt='torch_extension.matmul_bias(x, y, b, transC)',
        setup='',
        globals={'torch_extension': torch_extension, 'x': x, 'y': y, 'b': b, 'transC': transC},
        label=label,
        sub_label=sub_label,
        description = 'matmul_bias',
        ).blocked_autorange())

    compare = benchmark.Compare(results)
    print('\n')
    compare.print()
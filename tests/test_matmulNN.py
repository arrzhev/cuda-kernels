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

def torch_matmul_relu(x, y, b, use_relu):
    z_torch = x @ y + b
    z_torch_relu = z_torch.clamp(min=0) if use_relu else None
    return z_torch, z_torch_relu

LAYOUTS = ['tt', 'tn', 'nt', 'nn']
DTYPES = [torch.float32, torch.float16]
LIST_TF = [False, True]


IRREGULAR_SIZES = [(1, 1, 1), (1, 1, 1234), (1, 1234, 1), (1234, 1, 1), (10, 10, 10), (85, 77, 43), (123, 123, 123), (513, 512, 511)]
SCALE_SIZES = [(64, 64, 64), (256,256, 256), (512, 512, 512)]
SPECIAL_SIZES = [(32, 256, 784), (32, 128, 256), (32, 10, 128), (64, 256, 784), (64, 128, 256), (64, 10, 128)]

@pytest.mark.unit
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("transC", LIST_TF)
@pytest.mark.parametrize("use_relu", LIST_TF)
@pytest.mark.parametrize("M, N, K", IRREGULAR_SIZES + SCALE_SIZES + SPECIAL_SIZES)
def test_matmul_bias(dtype, layout, transC, use_relu, M, N, K):
    x, y = gen_mats(M, N, K, layout, dtype)
    b = torch.randn(N, dtype=dtype, device='cuda')

    z_torch, z_torch_relu = torch_matmul_relu(x, y, b, use_relu)
    z_extension, z_extension_relu = torch_extension.matmul_bias(x, y, b, use_relu, transC)

    torch.testing.assert_close(z_torch, z_extension, atol=1e-2, rtol=1e-2)
    if use_relu:
        torch.testing.assert_close(z_torch_relu, z_extension_relu, atol=1e-2, rtol=1e-2)

@pytest.mark.unit
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("transC", LIST_TF)
@pytest.mark.parametrize("M, N, K", IRREGULAR_SIZES + SCALE_SIZES + SPECIAL_SIZES)
def test_matmul_relu(dtype, layout, transC, M, N, K):
    x, y = gen_mats(M, N, K, layout, dtype)
    xr, _ = gen_mats(M, N, K, layout, dtype)

    z_torch = (x * (xr > 0)) @ y
    z_extension = torch_extension.matmul_relu(x, xr, y, transC)

    torch.testing.assert_close(z_torch, z_extension, atol=1e-2, rtol=1e-2)

@pytest.mark.performance
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("transC", LIST_TF)
@pytest.mark.parametrize("use_relu", LIST_TF)
def test_perf_matmul_bias(dtype, layout, transC, use_relu):
    results = []

    for M, N, K in IRREGULAR_SIZES + SCALE_SIZES + SPECIAL_SIZES:
        label = f'Matrix Mul with Bias {"and ReLU" if use_relu else ""} {str(dtype)} {layout} {"CT" if transC else ""}'
        sub_label = f'Matrix1: {M}x{K}; Matrix2: {K}x{N}'
        x, y = gen_mats(M, N, K, layout, dtype)
        b = torch.randn(N, dtype=dtype, device='cuda')

        results.append(benchmark.Timer(
            stmt='torch_matmul_relu(x, y, b, use_relu)',
            setup='',
            globals={'torch_matmul_relu': torch_matmul_relu,'x': x, 'y': y, 'b': b, 'use_relu': use_relu},
            label=label,
            sub_label=sub_label,
            description='torch',
        ).blocked_autorange())

        results.append(benchmark.Timer(
        stmt='torch_extension.matmul_bias(x, y, b, use_relu, transC)',
        setup='',
        globals={'torch_extension': torch_extension, 'x': x, 'y': y, 'b': b, 'use_relu': use_relu, 'transC': transC},
        label=label,
        sub_label=sub_label,
        description = 'matmul_bias',
        ).blocked_autorange())

    compare = benchmark.Compare(results)
    print('\n')
    compare.print()

@pytest.mark.performance
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("transC", LIST_TF)
def test_perf_matmul_relu(dtype, layout, transC):
    results = []

    for M, N, K in IRREGULAR_SIZES + SCALE_SIZES + SPECIAL_SIZES:
        label = f'Matrix Mul and ReLU {str(dtype)} {layout} {"CT" if transC else ""}'
        sub_label = f'Matrix1: {M}x{K}; Matrix2: {K}x{N}'
        x, y = gen_mats(M, N, K, layout, dtype)
        xr, _ = gen_mats(M, N, K, layout, dtype)

        results.append(benchmark.Timer(
            stmt='(x * (xr > 0)) @ y',
            setup='',
            globals={'x': x, 'xr': xr, 'y': y},
            label=label,
            sub_label=sub_label,
            description='torch',
        ).blocked_autorange())

        results.append(benchmark.Timer(
        stmt='torch_extension.matmul_relu(x, xr, y, transC)',
        setup='',
        globals={'torch_extension': torch_extension, 'x': x, 'xr': xr, 'y': y, 'transC': transC},
        label=label,
        sub_label=sub_label,
        description = 'matmul_relu',
        ).blocked_autorange())

    compare = benchmark.Compare(results)
    print('\n')
    compare.print()
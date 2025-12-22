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

def torch_matrix_reduction_row_relu(A, AR):
    dA = A * (AR > 0)
    return dA.sum(dim=0)

IRREGULAR_SIZES = [(1, 1), (1, 1234), (1234, 1), (10, 10), (85, 77), (123, 123), (513, 512), (512, 64)]
SCALE_SIZES = [(64, 64), (256,256), (512, 512)]
SPECIAL_SIZES = [(32, 256), (32, 128), (32, 10), (64, 256), (64, 128), (64, 10)]

@pytest.mark.unit
@pytest.mark.parametrize("M, N", IRREGULAR_SIZES + SCALE_SIZES + SPECIAL_SIZES)
def test_matrix_reduction_row(M, N):
    A = torch.randn(M, N, dtype=torch.float32, device='cuda')

    z_torch = A.sum(dim=0)
    z_extension = torch_extension.matrix_reduction_row(A)

    torch.testing.assert_close(z_torch, z_extension, atol=1e-2, rtol=1e-2)

@pytest.mark.unit
@pytest.mark.parametrize("M, N", IRREGULAR_SIZES + SCALE_SIZES + SPECIAL_SIZES)
def test_matrix_reduction_row_relu(M, N):
    A = torch.randn(M, N, dtype=torch.float32, device='cuda')
    AR = torch.randn(M, N, dtype=torch.float32, device='cuda')
    
    z_torch = torch_matrix_reduction_row_relu(A, AR)
    z_extension = torch_extension.matrix_reduction_row_relu(A, AR)

    torch.testing.assert_close(z_torch, z_extension, atol=1e-2, rtol=1e-2)

@pytest.mark.performance
def test_perf_matrix_reduction_row():
    results = []

    for M, N in IRREGULAR_SIZES + SCALE_SIZES + SPECIAL_SIZES:
        label = f'Matrix row reduction'
        sub_label = f'Matrix: {M}x{N}'
        A = torch.randn(M, N, dtype=torch.float32, device='cuda')
        AR = torch.randn(M, N, dtype=torch.float32, device='cuda')

        results.append(benchmark.Timer(
            stmt='A.sum(dim=0)',
            setup='',
            globals={'A': A},
            label=label,
            sub_label=sub_label,
            description='torch',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_matrix_reduction_row_relu(A, AR)',
            setup='',
            globals={'torch_matrix_reduction_row_relu': torch_matrix_reduction_row_relu,'A': A, 'AR': AR},
            label=label,
            sub_label=sub_label,
            description='torch relu',
        ).blocked_autorange())

        results.append(benchmark.Timer(
        stmt='torch_extension.matrix_reduction_row(A)',
        setup='',
        globals={'torch_extension': torch_extension, 'A': A},
        label=label,
        sub_label=sub_label,
        description = 'matrix_reduction_row',
        ).blocked_autorange())

        results.append(benchmark.Timer(
        stmt='torch_extension.matrix_reduction_row_relu(A, AR)',
        setup='',
        globals={'torch_extension': torch_extension, 'A': A, 'AR': AR},
        label=label,
        sub_label=sub_label,
        description = 'matrix_reduction_row_relu',
        ).blocked_autorange())

    compare = benchmark.Compare(results)
    print('\n')
    compare.print()
import pytest
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

import torch_extension

IRREGULAR_SIZES = [(1, 1), (1, 1234), (1234, 1), (10, 10), (85, 77), (123, 123), (513, 512), (512, 64)]
SCALE_SIZES = [(64, 64), (256,256), (512, 512)]

@pytest.mark.unit
@pytest.mark.parametrize("M, D", IRREGULAR_SIZES + SCALE_SIZES)
def test_layerNorm(M, D):
    X = torch.randn(M, D, dtype=torch.float32, device='cuda')
    W = torch.randn(D, dtype=torch.float32, device='cuda')
    B = torch.randn(D, dtype=torch.float32, device='cuda')

    y_torch = F.layer_norm(X, (D,), W, B, eps=1e-5)
    y_extension = torch_extension.layer_norm(X, W, B)

    torch.testing.assert_close(y_torch, y_extension, atol=1e-2, rtol=1e-2)

@pytest.mark.unit
@pytest.mark.parametrize("M, D", IRREGULAR_SIZES + SCALE_SIZES)
def test_RMSNorm(M, D):
    X = torch.randn(M, D, dtype=torch.float32, device='cuda')
    W = torch.randn(D, dtype=torch.float32, device='cuda')

    y_torch = F.rms_norm(X, (D,), W, eps=1e-5)
    y_extension = torch_extension.rms_norm(X, W)

    torch.testing.assert_close(y_torch, y_extension, atol=1e-2, rtol=1e-2)

@pytest.mark.performance
def test_perf_layer_norm():
    results = []

    for M, D in IRREGULAR_SIZES + SCALE_SIZES:
        label = f'Layer Norm'
        sub_label = f'Size: {M}x{D}'
        X = torch.randn(M, D, dtype=torch.float32, device='cuda')
        W = torch.randn(D, dtype=torch.float32, device='cuda')
        B = torch.randn(D, dtype=torch.float32, device='cuda')

        results.append(benchmark.Timer(
            stmt='F.layer_norm(X, (D,), W, B)',
            setup='',
            globals={'F': torch.nn.functional, 'X': X, 'W': W, 'B': B, 'D': D},
            label=label,
            sub_label=sub_label,
            description='torch',
        ).blocked_autorange())


        results.append(benchmark.Timer(
        stmt='torch_extension.layer_norm(X, W, B)',
        setup='',
        globals={'torch_extension': torch_extension, 'X': X, 'W': W, 'B': B},
        label=label,
        sub_label=sub_label,
        description = 'layer_norm',
        ).blocked_autorange())

    compare = benchmark.Compare(results)
    print('\n')
    compare.print()

@pytest.mark.performance
def test_perf_rms_norm():
    results = []

    for M, D in IRREGULAR_SIZES + SCALE_SIZES:
        label = f'RMS Norm'
        sub_label = f'Size: {M}x{D}'
        X = torch.randn(M, D, dtype=torch.float32, device='cuda')
        W = torch.randn(D, dtype=torch.float32, device='cuda')

        results.append(benchmark.Timer(
            stmt='F.rms_norm(X, (D,), W, eps=1e-5)',
            setup='',
            globals={'F': torch.nn.functional, 'X': X, 'W': W, 'D': D},
            label=label,
            sub_label=sub_label,
            description='torch',
        ).blocked_autorange())


        results.append(benchmark.Timer(
        stmt='torch_extension.rms_norm(X, W)',
        setup='',
        globals={'torch_extension': torch_extension, 'X': X, 'W': W},
        label=label,
        sub_label=sub_label,
        description = 'rms_norm',
        ).blocked_autorange())

    compare = benchmark.Compare(results)
    print('\n')
    compare.print()
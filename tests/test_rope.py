import pytest
import torch
import torch.utils.benchmark as benchmark

import torch_extension

def get_sin_cos(seq_len, dim, dtype=torch.float32, device='cuda'):

    half = dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=dtype, device=device) / half))
    positions = torch.arange(seq_len, dtype=dtype, device=device)
    theta = positions[:, None] * inv_freq[None, :]
    sin_cos = torch.cat((torch.sin(theta).unsqueeze(-1), torch.cos(theta).unsqueeze(-1)), dim=-1)

    return sin_cos

def rope(x):
    b, seq_len, dim = x.shape
    assert dim % 2 == 0, "Dimension must be even for RoPE"

    # compute pair frequencies
    half = dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=x.dtype, device=x.device) / half))
    positions = torch.arange(seq_len, dtype=x.dtype, device=x.device)

    # (seq_len, half)
    theta = positions[:, None] * inv_freq[None, :]
    sin = torch.sin(theta)  # (seq_len, half)
    cos = torch.cos(theta)  # (seq_len, half)

    # split x
    x_even = x[..., 0::2]  # (batch, seq_len, half)
    x_odd = x[..., 1::2]

    # apply RoPE
    rot_even = x_even * cos.unsqueeze(0) - x_odd * sin.unsqueeze(0)
    rot_odd  = x_even * sin.unsqueeze(0) + x_odd * cos.unsqueeze(0)

    # interleave back
    y = torch.empty_like(x)
    y[..., 0::2] = rot_even
    y[..., 1::2] = rot_odd
    return y


IRREGULAR_SIZES = [(1, 1, 2), (1, 1, 1234), (1, 1234, 2), (1234, 1, 2), (10, 10, 10), (85, 77, 44), (123, 123, 124)]
SCALE_SIZES = [(64, 64, 64), (256, 256, 256)]

@pytest.mark.unit
@pytest.mark.parametrize("B, L, D", IRREGULAR_SIZES + SCALE_SIZES)
def test_rope(B, L, D):
    X = torch.randn(B, L, D, dtype=torch.float32, device='cuda')

    y_extension = torch_extension.rope(X)
    y_torch = rope(X)

    torch.testing.assert_close(y_torch, y_extension, atol=1e-2, rtol=1e-2)

@pytest.mark.unit
@pytest.mark.parametrize("B, L, D", IRREGULAR_SIZES + SCALE_SIZES)
def test_rope_cached(B, L, D):
    X = torch.randn(B, L, D, dtype=torch.float32, device='cuda')

    sin_cos = get_sin_cos(L, D, X.dtype, X.device)

    y_extension = torch_extension.rope_cached(X, sin_cos)
    y_torch = rope(X)

    torch.testing.assert_close(y_torch, y_extension, atol=1e-2, rtol=1e-2)

@pytest.mark.performance
def test_perf_rope():
    results = []

    for B, L, D in IRREGULAR_SIZES + SCALE_SIZES:
        label = f'ROPE'
        sub_label = f'Size: {B}x{L}x{D}'
        X = torch.randn(B, L, D, dtype=torch.float32, device='cuda')
        sin_cos = get_sin_cos(L, D, X.dtype, X.device)

        results.append(benchmark.Timer(
            stmt='rope(X)',
            setup='',
            globals={'rope': rope, 'X': X},
            label=label,
            sub_label=sub_label,
            description='torch',
        ).blocked_autorange())


        results.append(benchmark.Timer(
        stmt='torch_extension.rope(X)',
        setup='',
        globals={'torch_extension': torch_extension, 'X': X},
        label=label,
        sub_label=sub_label,
        description = 'rope',
        ).blocked_autorange())

        results.append(benchmark.Timer(
        stmt='torch_extension.rope_cached(X, sin_cos)',
        setup='',
        globals={'torch_extension': torch_extension, 'X': X, 'sin_cos': sin_cos},
        label=label,
        sub_label=sub_label,
        description = 'rope_cached',
        ).blocked_autorange())

    compare = benchmark.Compare(results)
    print('\n')
    compare.print()
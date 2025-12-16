import pytest
import torch
import torch.utils.benchmark as benchmark
import torch.nn.functional as F
import itertools

import torch_extension

def mean_blur(image, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd integer for symmetric padding.")

    padding = kernel_size // 2

    blurred = F.avg_pool2d(image.float(), kernel_size=kernel_size, stride=1, padding=padding, count_include_pad=False)

    return blurred.byte()

@pytest.mark.unit
@pytest.mark.parametrize("size", [1, 10, 256, 1213, 4096])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("kernel_size", [1, 5])
def test_meanBlur(size, channels, kernel_size):
    image = torch.randint(0, 256, (channels, size, size), dtype=torch.uint8, device="cuda")

    blurred_torch = mean_blur(image, kernel_size)
    blurred_extension = torch_extension.mean_blur(image, kernel_size)

    torch.testing.assert_close(blurred_torch, blurred_extension)

@pytest.mark.performance
def test_perf_meanBlur():
    results = []
    sizes = [1, 10, 256, 1213, 4096]
    channels = [1 ,3]
    kernel_sizes =  [1, 5]

    for size, channel, kernel_size  in itertools.product(sizes, channels, kernel_sizes):
        label = 'Mean blur'
        sub_label = f'size: {size}x{size}; channel: {channel}; kernel: {kernel_size}'

        image = torch.randint(0, 256, (channel, size, size), dtype=torch.uint8, device="cuda")

        results.append(benchmark.Timer(
            stmt='mean_blur(image, kernel_size)',
            setup='',
            globals={'mean_blur': mean_blur, 'image': image, 'kernel_size': kernel_size},
            label=label,
            sub_label=sub_label,
            description='torch',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.mean_blur(image, kernel_size)',
            setup='import torch_extension',
            globals={'image': image, 'kernel_size': kernel_size},
            label=label,
            sub_label=sub_label,
            description='extension',
        ).blocked_autorange())

    compare = benchmark.Compare(results)
    compare.print()
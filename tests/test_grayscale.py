import pytest
import torch
import torch.utils.benchmark as benchmark
import torchvision.transforms as transforms

import torch_extension

@pytest.mark.unit
@pytest.mark.parametrize("size", [1, 10, 256, 1213, 4096, 8000])
def test_rgb2gray(size):
    image_HWC = torch.randint(0, 256, (size, size, 3), dtype=torch.uint8, device="cuda")
    image_CHW_strided = image_HWC.permute(2, 0, 1)
    image_CHW = image_CHW_strided.contiguous()
    image_HWC_strided = image_CHW.permute(1, 2, 0)

    grayscale_transform = transforms.Grayscale()
    grayscale_planar_torch = grayscale_transform(image_CHW_strided)
    grayscale_interleaved_torch = grayscale_planar_torch.permute(1, 2, 0)

    grayscale_interleaved_contiguous_extension = torch_extension.rgb2gray(image_HWC)
    grayscale_interleaved_strided_extension = torch_extension.rgb2gray(image_HWC_strided)

    grayscale_planar_contiguous_extension = torch_extension.rgb2gray(image_CHW)
    grayscale_planar_strided_extension = torch_extension.rgb2gray(image_CHW_strided)

    torch.testing.assert_close(grayscale_interleaved_torch, grayscale_interleaved_contiguous_extension, atol=1, rtol=0)
    torch.testing.assert_close(grayscale_interleaved_torch, grayscale_interleaved_strided_extension, atol=1, rtol=0)

    torch.testing.assert_close(grayscale_planar_torch, grayscale_planar_contiguous_extension, atol=1, rtol=0)
    torch.testing.assert_close(grayscale_planar_torch, grayscale_planar_strided_extension, atol=1, rtol=0)

@pytest.mark.performance
def test_perf_rgb2gray():
    results = []
    sizes = [1, 10, 256, 1213, 4096, 8000, 8001]
    grayscale_transform = transforms.Grayscale()

    for size in sizes:
        label = 'Grayscale'
        sub_label = f'size: {size}x{size}'

        image_HWC = torch.randint(0, 256, (size, size, 3), dtype=torch.uint8, device="cuda")
        image_CHW_strided = image_HWC.permute(2, 0, 1)
        image_CHW = image_CHW_strided.contiguous()
        image_HWC_strided = image_CHW.permute(1, 2, 0)

        results.append(benchmark.Timer(
            stmt='grayscale_transform(image_CHW)',
            setup='',
            globals={'grayscale_transform': grayscale_transform, 'image_CHW': image_CHW},
            label=label,
            sub_label=sub_label,
            description='torch CHW cont',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='grayscale_transform(image_CHW_strided)',
            setup='',
            globals={'grayscale_transform': grayscale_transform, 'image_CHW_strided': image_CHW_strided},
            label=label,
            sub_label=sub_label,
            description='torch CHW strided',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.rgb2gray(image_HWC)',
            setup='import torch_extension',
            globals={'image_HWC': image_HWC},
            label=label,
            sub_label=sub_label,
            description='ext HWC cont',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.rgb2gray(image_HWC_strided)',
            setup='import torch_extension',
            globals={'image_HWC_strided': image_HWC_strided},
            label=label,
            sub_label=sub_label,
            description='ext HWC strided',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.rgb2gray(image_CHW)',
            setup='import torch_extension',
            globals={'image_CHW': image_CHW},
            label=label,
            sub_label=sub_label,
            description='ext CHW cont',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.rgb2gray(image_CHW_strided)',
            setup='import torch_extension',
            globals={'image_CHW_strided': image_CHW_strided},
            label=label,
            sub_label=sub_label,
            description='ext CHW strided',
        ).blocked_autorange())

    compare = benchmark.Compare(results)
    compare.print()
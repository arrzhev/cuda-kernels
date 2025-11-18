import pytest
import torch
import torch.utils.benchmark as benchmark
import torchvision.transforms as transforms
import skimage
import matplotlib.pyplot as plt

import torch_extension

@pytest.mark.unit
@pytest.mark.parametrize("size", [1, 10, 256, 1213, 4096, 8000])
def test_rgb2gray(size):
    image_HWC = torch.randint(0, 256, (size, size, 3), dtype=torch.uint8, device="cuda")
    image_CHW = image_HWC.permute(2, 0, 1)

    grayscale_transform = transforms.Grayscale()
    grayscale_planar_torch = grayscale_transform(image_CHW)
    grayscale_interleaved_torch = grayscale_planar_torch.permute(1, 2, 0)

    grayscale_interleaved_extension = torch_extension.rgb2gray(image_HWC)
    grayscale_planar_extension = torch_extension.rgb2gray(image_CHW)

    torch.testing.assert_close(grayscale_interleaved_torch, grayscale_interleaved_extension, atol=1, rtol=0)
    torch.testing.assert_close(grayscale_planar_torch, grayscale_planar_extension, atol=1, rtol=0)

@pytest.mark.performance
def test_perf_rgb2gray():
    results = []
    sizes = [1, 10, 256, 1213, 4096, 8000]
    grayscale_transform = transforms.Grayscale()

    for size in sizes:
        label = 'Grayscale'
        sub_label = f'size: {size}x{size}'

        image_HWC = torch.randint(0, 256, (size, size, 3), dtype=torch.uint8, device="cuda")
        image_CHW = image_HWC.permute(2, 0, 1)

        results.append(benchmark.Timer(
            stmt='grayscale_transform(image_CHW)',
            setup='',
            globals={'grayscale_transform': grayscale_transform, 'image_CHW': image_CHW},
            label=label,
            sub_label=sub_label,
            description='torch',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.rgb2gray(image_HWC)',
            setup='import torch_extension',
            globals={'image_HWC': image_HWC},
            label=label,
            sub_label=sub_label,
            description='extension interleaved contiguous',
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch_extension.rgb2gray(image_CHW)',
            setup='import torch_extension',
            globals={'image_CHW': image_CHW},
            label=label,
            sub_label=sub_label,
            description='extension planar strided',
        ).blocked_autorange())

    compare = benchmark.Compare(results)
    compare.print()

if __name__ == '__main__':
    image_HWC = torch.from_numpy(skimage.data.astronaut()).cuda()
    image_CHW = image_HWC.permute(2, 0, 1)
    
    grayscale_transform = transforms.Grayscale()
    grayscale_planar_torch = grayscale_transform(image_CHW)

    grayscale_interleaved_extension = torch_extension.rgb2gray(image_HWC)
    grayscale_planar_extension = torch_extension.rgb2gray(image_CHW)

    _, axes = plt.subplots(nrows=2, ncols=2)

    images = [
        {'image': image_HWC.cpu(), 'cmap': None, 'title': 'Original image'},
        {'image': grayscale_planar_torch.cpu().squeeze().numpy(), 'cmap': 'gray', 'title': 'PyTorch grayscale'},
        {'image': grayscale_interleaved_extension.cpu().squeeze().numpy(), 'cmap': 'gray', 'title': 'My grayscale Interleaved'},
        {'image': grayscale_planar_extension.cpu().squeeze().numpy(), 'cmap': 'gray', 'title': 'My grayscale Planar'},
    ]

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i]['image'], images[i]['cmap'])
        ax.set_title(images[i]['title'])
        ax.axis('off')

    plt.tight_layout()
    plt.show()
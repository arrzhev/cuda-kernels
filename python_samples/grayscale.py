import torch
import torchvision.transforms as transforms
import skimage
import matplotlib.pyplot as plt

import torch_extension

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
import torchvision
import torch.utils.data as data
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import numpy as np
import torch

from ops.dataset import TSNDataSet
from ops.transforms import *

dataset = 'ssv2'
if dataset == 'minikinetics':
    root_path = '/data1/minikinetics/frames/images'
    val_list = '/data1/minikinetics/labels/val_videofolder.txt'
    prefix = 'image_{:05d}.jpg'
elif dataset == 'ssv2':
    root_path = '/data2/sthsthv2/tsm_img'
    val_list = '/data2/sthsthv2/val_videofolder.txt'
    prefix = '{:06d}.jpg'
num_segments = 8
alpha = 100

def one_hot(x, count):
    return torch.eye(count)[x, :]

if dataset == 'minikinetics':
    val_loader = torch.utils.data.DataLoader(
                TSNDataSet(dataset, root_path + '/val/', val_list, num_segments=num_segments,
                           modality='RGB',
                           image_tmpl=prefix,
                           transform=torchvision.transforms.Compose([
                               Stack(),
                               ToTorchFormatTensor(div=False),
                           ]),
                           dense_sample=False),
                batch_size=1, shuffle=False,
                num_workers=8, pin_memory=True)
else:
    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(dataset, root_path, val_list, num_segments=num_segments,
                   modality='RGB',
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       Stack(),
                       ToTorchFormatTensor(div=False),
                   ]),
                   dense_sample=False),
        batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True)

for index, (input, target) in enumerate(val_loader):
    height = input.shape[2]
    width = input.shape[3]
    input = input.squeeze()

    # TODO: RGBRGBRGB... -> RRR...GGG...BBB...
    r_indices = (torch.arange(num_segments) * 3)
    input_r = torch.index_select(input, 0, r_indices)
    input_g = torch.index_select(input, 0, r_indices + 1)
    input_b = torch.index_select(input, 0, r_indices + 2)

    # TODO: flip
    # input_r_flipped = torch.flip(input_r, [0])
    # input_g_flipped = torch.flip(input_g, [0])
    # input_b_flipped = torch.flip(input_b, [0])

    # TODO: FFT
    input_r_fft = torch.fft.fftn(input_r, dim=[0, 1, 2])
    input_g_fft = torch.fft.fftn(input_g, dim=[0, 1, 2])
    input_b_fft = torch.fft.fftn(input_b, dim=[0, 1, 2])

    # TODO: Phase-only
    input_r_phase = torch.fft.ifftn(torch.ones_like(input_r_fft.abs()) * torch.exp(1j * input_r_fft.angle()), dim=[0, 1, 2])
    input_g_phase = torch.fft.ifftn(torch.ones_like(input_g_fft.abs()) * torch.exp(1j * input_g_fft.angle()), dim=[0, 1, 2])
    input_b_phase = torch.fft.ifftn(torch.ones_like(input_b_fft.abs()) * torch.exp(1j * input_b_fft.angle()), dim=[0, 1, 2])
    input_r_phase *= input_r_fft.abs()
    input_g_phase *= input_g_fft.abs()
    input_b_phase *= input_b_fft.abs()

    # TODO: RRR...GGG...BBB... -> RGBRGBRGB...
    input_reshape = torch.cat((input_r_phase, input_g_phase, input_b_phase), dim=0)

    input_new = torch.zeros_like(input)
    for i in range(3 * num_segments):
        if i < num_segments:
            input_new[3 * i, :, :] = input_reshape[i, :, :]
        elif i < 2 * num_segments:
            input_new[3 * (i - num_segments) + 1, :, :] = input_reshape[i, :, :]
        else:
            input_new[3 * (i - 2 * num_segments) + 2, :, :] = input_reshape[i, :, :]

    # TODO: Show images
    input = input.reshape(num_segments, 3, height, width)
    input_new = input_new.reshape(num_segments, 3, height, width)

    if index % 10 == 0:
        fig = plt.figure()
        rows = 2
        cols = num_segments
        for i in range(num_segments):
            ax1 = fig.add_subplot(rows, cols, 1 + i)
            ax1.imshow(np.asarray(input[i].permute(1, 2, 0), dtype=np.uint8))
            ax1.axis('off')
            if i == 0:
                ax1.set_title("Original")

            ax2 = fig.add_subplot(rows, cols, num_segments + 1 + i)
            ax2.imshow(np.asarray(input_new[i].permute(1, 2, 0), dtype=np.uint8))
            ax2.axis('off')
            if i == 0:
                ax2.set_title("Phase Only")

        plt.show()



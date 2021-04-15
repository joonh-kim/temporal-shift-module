import torch.utils.data as data
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import numpy as np

from ops.dataset import TSNDataSet
from ops.transforms import *
from ops.utils import DCTmatrix, DCTmatrix_rgb

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
modality = 'Time domain' ## 'Time domain' or 'Frquency domain'
lowest = True
if lowest:
    sorting = 'low frequencies'
else:
    sorting = 'high frequencies'
percentage = [75, 50, 25]


DCT = DCTmatrix(num_segments)
DCT_hat = DCTmatrix_rgb(DCT, num_segments)
DCT, DCT_hat = torch.from_numpy(DCT).type(torch.float32), torch.from_numpy(DCT_hat).type(torch.float32)


def one_hot(x, count):
    return torch.eye(count)[x, :]

if dataset == 'minikinetics':
    val_loader = torch.utils.data.DataLoader(
                TSNDataSet(root_path + '/val/', val_list, num_segments=num_segments,
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
        TSNDataSet(root_path, val_list, num_segments=num_segments,
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
    input_r = torch.index_select(input.reshape(3 * num_segments, -1), 0, r_indices)
    input_g = torch.index_select(input.reshape(3 * num_segments, -1), 0, r_indices + 1)
    input_b = torch.index_select(input.reshape(3 * num_segments, -1), 0, r_indices + 2)

    # TODO: flip
    input_r_flipped = torch.flip(input_r, [0])
    input_g_flipped = torch.flip(input_g, [0])
    input_b_flipped = torch.flip(input_b, [0])

    input_reshape = torch.cat((input_r, input_g, input_b), dim=0)
    input_reshape_flipped = torch.cat((input_r_flipped, input_g_flipped, input_b_flipped), dim=0)

    # TODO: DCT
    input_DCT = torch.matmul(DCT_hat, input_reshape)
    input_DCT_flipped = torch.matmul(DCT_hat, input_reshape_flipped)

    # TODO: Select salient frequencies
    freq_saliency = torch.zeros(num_segments, height * width)
    for i in range(num_segments):
        freq_saliency[i, :] = input_DCT[i, :] + input_DCT[num_segments + i, :] + input_DCT[2 * num_segments + i, :]
    _, indices_h = torch.topk(torch.norm(freq_saliency, dim=1), k=int(num_segments * percentage[0]/100))
    _, indices_i = torch.topk(torch.norm(freq_saliency, dim=1), k=int(num_segments * percentage[1]/100))
    _, indices_l = torch.topk(torch.norm(freq_saliency, dim=1), k=int(num_segments * percentage[2]/100))

    mask_h = one_hot(indices_h, num_segments).sum(dim=0) if lowest else 1 - one_hot(indices_l, num_segments).sum(dim=0)
    mask_i = one_hot(indices_i, num_segments).sum(dim=0) if lowest else 1 - one_hot(indices_i, num_segments).sum(dim=0)
    mask_l = one_hot(indices_l, num_segments).sum(dim=0) if lowest else 1 - one_hot(indices_h, num_segments).sum(dim=0)

    mask_h_exp = mask_h.repeat(1, 3)
    mask_i_exp = mask_i.repeat(1, 3)
    mask_l_exp = mask_l.repeat(1, 3)

    masked_h_hat = mask_h_exp.squeeze().unsqueeze(-1) * DCT_hat
    masked_i_hat = mask_i_exp.squeeze().unsqueeze(-1) * DCT_hat
    masked_l_hat = mask_l_exp.squeeze().unsqueeze(-1) * DCT_hat
    if modality == 'Time domain':
        input_masked_h = torch.matmul(torch.transpose(masked_h_hat, 0, 1), torch.matmul(masked_h_hat, input_reshape))
        input_masked_i = torch.matmul(torch.transpose(masked_i_hat, 0, 1), torch.matmul(masked_i_hat, input_reshape))
        input_masked_l = torch.matmul(torch.transpose(masked_l_hat, 0, 1), torch.matmul(masked_l_hat, input_reshape))
    elif modality == 'Frequency domain':
        input_masked_h = torch.matmul(masked_h_hat, input_reshape)
        input_masked_i = torch.matmul(masked_i_hat, input_reshape)
        input_masked_l = torch.matmul(masked_l_hat, input_reshape)

    # TODO: RRR...GGG...BBB... -> RGBRGBRGB...
    input_new_a = torch.zeros_like(input_DCT)
    for i in range(3 * num_segments):
        if i < num_segments:
            input_new_a[3 * i, :] = input_DCT[i, :]
        elif i < 2 * num_segments:
            input_new_a[3 * (i - num_segments) + 1, :] = input_DCT[i, :]
        else:
            input_new_a[3 * (i - 2 * num_segments) + 2, :] = input_DCT[i, :]

    input_new_flipped = torch.zeros_like(input_DCT_flipped)
    for i in range(3 * num_segments):
        if i < num_segments:
            input_new_flipped[3 * i, :] = input_DCT_flipped[i, :]
        elif i < 2 * num_segments:
            input_new_flipped[3 * (i - num_segments) + 1, :] = input_DCT_flipped[i, :]
        else:
            input_new_flipped[3 * (i - 2 * num_segments) + 2, :] = input_DCT_flipped[i, :]

    input_new_h = torch.zeros_like(input_masked_h)
    for i in range(3 * num_segments):
        if i < num_segments:
            input_new_h[3 * i, :] = input_masked_h[i, :]
        elif i < 2 * num_segments:
            input_new_h[3 * (i - num_segments) + 1, :] = input_masked_h[i, :]
        else:
            input_new_h[3 * (i - 2 * num_segments) + 2, :] = input_masked_h[i, :]

    input_new_i = torch.zeros_like(input_masked_i)
    for i in range(3 * num_segments):
        if i < num_segments:
            input_new_i[3 * i, :] = input_masked_i[i, :]
        elif i < 2 * num_segments:
            input_new_i[3 * (i - num_segments) + 1, :] = input_masked_i[i, :]
        else:
            input_new_i[3 * (i - 2 * num_segments) + 2, :] = input_masked_i[i, :]

    input_new_l = torch.zeros_like(input_masked_l)
    for i in range(3 * num_segments):
        if i < num_segments:
            input_new_l[3 * i, :] = input_masked_l[i, :]
        elif i < 2 * num_segments:
            input_new_l[3 * (i - num_segments) + 1, :] = input_masked_l[i, :]
        else:
            input_new_l[3 * (i - 2 * num_segments) + 2, :] = input_masked_l[i, :]


    # TODO: Show images
    input = input.reshape(num_segments, 3, height, width)
    input_new_a = input_new_a.reshape(num_segments, 3, height, width)
    input_new_flipped = input_new_flipped.reshape(num_segments, 3, height, width)
    input_new_h = input_new_h.reshape(num_segments, 3, height, width)
    input_new_i = input_new_i.reshape(num_segments, 3, height, width)
    input_new_l = input_new_l.reshape(num_segments, 3, height, width)
    for i in range(num_segments):
        for j in range(3):
            input_new_a[i, j, :] = (input_new_a[i, j, :] - torch.mean(input_new_a[i, j, :])) / torch.std(input_new_a[i, j, :])
            input_new_flipped[i, j, :] = (input_new_flipped[i, j, :] - torch.mean(input_new_flipped[i, j, :])) / torch.std(input_new_flipped[i, j, :])

    if modality == 'Freq':
        for i in range(num_segments):
            for j in range(3):
                input_new_h[i, j, :] = (input_new_h[i, j, :] - torch.mean(input_new_h[i, j, :])) / torch.std(input_new_h[i, j, :])
                input_new_i[i, j, :] = (input_new_i[i, j, :] - torch.mean(input_new_i[i, j, :])) / torch.std(input_new_i[i, j, :])
                input_new_l[i, j, :] = (input_new_l[i, j, :] - torch.mean(input_new_l[i, j, :])) / torch.std(input_new_l[i, j, :])

    if index % 10 == 0:
        fig = plt.figure()
        rows = 6
        cols = num_segments
        for i in range(num_segments):
            ax1 = fig.add_subplot(rows, cols, 1 + i)
            ax1.imshow(np.asarray(input[i].permute(1, 2, 0), dtype=np.uint8))
            ax1.axis('off')
            if i == 0:
                ax1.set_title("Original")

            ax2 = fig.add_subplot(rows, cols, num_segments + 1 + i)
            ax2.imshow(np.asarray(input_new_a[i].permute(1, 2, 0), dtype=np.uint8))
            ax2.axis('off')
            if i == 0:
                ax2.set_title("Frequency domain")

            ax3 = fig.add_subplot(rows, cols, 2 * num_segments + 1 + i)
            ax3.imshow(np.asarray(input_new_h[i].permute(1, 2, 0), dtype=np.uint8))
            ax3.axis('off')
            if i == 0:
                ax3.set_title("{}% {} remained ({})".format(str(percentage[0]), sorting, modality))

            ax4 = fig.add_subplot(rows, cols, 3 * num_segments + 1 + i)
            ax4.imshow(np.asarray(input_new_i[i].permute(1, 2, 0), dtype=np.uint8))
            ax4.axis('off')
            if i == 0:
                ax4.set_title("{}% {} remained ({})".format(str(percentage[1]), sorting, modality))

            ax5 = fig.add_subplot(rows, cols, 4 * num_segments + 1 + i)
            ax5.imshow(np.asarray(input_new_l[i].permute(1, 2, 0), dtype=np.uint8))
            ax5.axis('off')
            if i == 0:
                ax5.set_title("{}% {} remained ({})".format(str(percentage[2]), sorting, modality))

            ax6 = fig.add_subplot(rows, cols, 5 * num_segments + 1 + i)
            ax6.imshow(np.asarray(input_new_flipped[i].permute(1, 2, 0), dtype=np.uint8))
            ax6.axis('off')
            if i == 0:
                ax6.set_title("Frequency domain (flipped)")

        plt.show()



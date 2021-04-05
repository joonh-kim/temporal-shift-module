import os
from PIL import Image
import torch
import numpy as np
from ops.utils import DCTmatrix
from ops.dataset_config import *

if __name__ == '__main__':
    dataset = 'minikinetics'
    modality = 'RGB'
    num_frames = 8

    DCT = DCTmatrix(num_frames)
    DCT = torch.from_numpy(DCT).cuda()

    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                   'kinetics': return_kinetics, 'minikinetics': return_minikinetics}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset ' + dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    with open(file_imglist_train) as f:
        lines = f.readlines()
    if dataset in ['minikinetics']:
        videos = [os.path.join(root_data, 'train', item.split(',')[0]) for item in lines]
    else:
        videos = [os.path.join(root_data, item.split(',')[0]) for item in lines]
    total_frames = [item.split(',')[1] for item in lines]

    rgb_freqs = []
    for i, (video, total_frame) in enumerate(zip(videos, total_frames)):
        if i % 1000 == 0:
            print("{}/{}".format(i, len(videos)))
        if not os.path.exists(os.path.join(video, prefix.format(int(total_frame.zfill(5))))):
            print("Not found: {}".format(os.path.join(video, prefix.format(int(total_frame.zfill(5))))))
            continue
        if int(total_frame) >= 2 * num_frames:
            indices = torch.arange(num_frames)
            iters = int(total_frame) // num_frames
            indices = indices * iters + 1
            rgb_freq = []
            for iter in range(iters):
                imgs = [
                    torch.from_numpy(np.array(
                        Image.open(os.path.join(video, prefix.format(int(str(int(indice)).zfill(5))))).convert(
                            'RGB'))).unsqueeze(-1) for indice in indices + iter]
                imgs = torch.cat(imgs, -1)
                imgs = imgs.cuda()
                imgs_r, imgs_g, imgs_b = \
                    imgs[:, :, 0, :].squeeze(), imgs[:, :, 1, :].squeeze(), imgs[:, :, 2, :].squeeze()
                imgs_r = torch.transpose(imgs_r.reshape(imgs_r.shape[0] * imgs_r.shape[1], -1), 0, 1)
                imgs_g = torch.transpose(imgs_g.reshape(imgs_g.shape[0] * imgs_g.shape[1], -1), 0, 1)
                imgs_b = torch.transpose(imgs_b.reshape(imgs_b.shape[0] * imgs_b.shape[1], -1), 0, 1)

                freqs_r, freqs_g, freqs_b = torch.matmul(DCT, imgs_r.type(torch.float64)), \
                                            torch.matmul(DCT, imgs_g.type(torch.float64)), \
                                            torch.matmul(DCT, imgs_b.type(torch.float64))
                rgb_freq.append(list(map(lambda x: torch.mean(x), [freqs_r, freqs_g, freqs_b])))
            rgb_freqs.append(torch.mean(torch.tensor(rgb_freq), dim=0))
        else:
            if int(total_frame) < num_frames:
                indices = torch.ones(num_frames)
            else:
                indices = torch.arange(num_frames) + 1
            imgs = [
                torch.from_numpy(np.array(
                    Image.open(os.path.join(video, prefix.format(int(str(int(indice)).zfill(5))))).convert(
                        'RGB'))).unsqueeze(-1) for indice in indices]
            imgs = torch.cat(imgs, -1)
            imgs = imgs.cuda()
            imgs_r, imgs_g, imgs_b = \
                imgs[:, :, 0, :].squeeze(), imgs[:, :, 1, :].squeeze(), imgs[:, :, 2, :].squeeze()
            imgs_r = torch.transpose(imgs_r.reshape(imgs_r.shape[0] * imgs_r.shape[1], -1), 0, 1)
            imgs_g = torch.transpose(imgs_g.reshape(imgs_g.shape[0] * imgs_g.shape[1], -1), 0, 1)
            imgs_b = torch.transpose(imgs_b.reshape(imgs_b.shape[0] * imgs_b.shape[1], -1), 0, 1)

            freqs_r, freqs_g, freqs_b = torch.matmul(DCT, imgs_r.type(torch.float64)), \
                                        torch.matmul(DCT, imgs_g.type(torch.float64)), \
                                        torch.matmul(DCT, imgs_b.type(torch.float64))
            rgb_freqs.append(torch.tensor(list(map(lambda x: torch.mean(x), [freqs_r, freqs_g, freqs_b]))))
        i += 1

    torch.save(torch.tensor(rgb_freqs), './freqs/freqs_{}_{}.pt'.format(dataset, num_frames))
    freqs_mean = torch.mean(torch.tensor(rgb_freqs), dim=0)
    freqs_std = torch.std(torch.tensor(rgb_freqs), dim=0)
    print("Mean (RGB): {}, Std (RGB): {}".format(freqs_mean, freqs_std))
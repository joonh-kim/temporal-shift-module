import os
from PIL import Image
import torch
import numpy as np
from ops.utils import DCTmatrix
from ops.dataset_config import *
import multiprocessing
from multiprocessing import Pool
import itertools
import parmap

def mean_and_std(video, prefix, DCT):
    item = video.split(',')
    video, total_frame = item[0], item[1]
    if not os.path.exists(os.path.join(video, prefix.format(int(total_frame.zfill(5))))):
        print("Not found: {}".format(os.path.join(video, prefix.format(int(total_frame.zfill(5))))))
        return np.array([None, None, None])
    else:
        if int(total_frame) >= 2 * num_frames:
            indices = np.arange(num_frames)
            iters = int(total_frame) // num_frames
            indices = indices * iters + 1
            rgb_freq = []
            for iter in range(iters):
                imgs = np.array([np.array(
                    Image.open(os.path.join(video, prefix.format(int(str(int(indice)).zfill(5))))).convert('RGB'))
                    for indice in indices + iter])

                imgs_r, imgs_g, imgs_b = imgs[:, :, :, 0], imgs[:, :, :, 1], imgs[:, :, :, 2]

                freqs_r, freqs_g, freqs_b = np.matmul(DCT, imgs_r.reshape((num_frames, -1))), \
                                            np.matmul(DCT, imgs_g.reshape((num_frames, -1))), \
                                            np.matmul(DCT, imgs_b.reshape((num_frames, -1)))
                rgb_freq.append(list(map(lambda x: np.mean(x), [freqs_r, freqs_g, freqs_b])))
            return np.mean(np.array(rgb_freq), axis=0)
        else:
            if int(total_frame) < num_frames:
                indices = np.ones(num_frames)
            else:
                indices = np.arange(num_frames) + 1
            imgs = np.array([np.array(
                Image.open(os.path.join(video, prefix.format(int(str(int(indice)).zfill(5))))).convert('RGB'))
                for indice in indices])

            imgs_r, imgs_g, imgs_b = imgs[:, :, :, 0], imgs[:, :, :, 1], imgs[:, :, :, 2]

            freqs_r, freqs_g, freqs_b = np.matmul(DCT, imgs_r.reshape((num_frames, -1))), \
                                        np.matmul(DCT, imgs_g.reshape((num_frames, -1))), \
                                        np.matmul(DCT, imgs_b.reshape((num_frames, -1)))
            return np.array(list(map(lambda x: torch.mean(x), [freqs_r, freqs_g, freqs_b])))

if __name__ == '__main__':
    dataset = 'minikinetics'
    modality = 'RGB'
    num_frames = 8
    # num_process = multiprocessing.cpu_count()
    num_process = 10

    DCT = DCTmatrix(num_frames)

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
        videos = [os.path.join(root_data, 'train', item.split(',')[0]) + ',' + item.split(',')[1]
                  for i, item in enumerate(lines)]
    else:
        videos = [os.path.join(root_data, item.split(',')[0]) + ',' + item.split(',')[1]
                  for i, item in enumerate(lines)]

    # with Pool(num_process) as p:
    #     rgb_freqs = np.array(list(p.starmap(mean_and_std, zip(videos, itertools.repeat(prefix), itertools.repeat(DCT))))
    #                          , dtype=np.float64)
    rgb_freqs = np.array(list(parmap.starmap(mean_and_std, list(zip(videos, itertools.repeat(prefix), itertools.repeat(DCT))),
                                             pm_pbar=True, pm_processes=num_process))
                         , dtype=np.float64)

    freqs_mean = np.nanmean(rgb_freqs, axis=0)
    freqs_std = np.nanstd(rgb_freqs, axis=0)
    print("Mean (RGB): {}, Std (RGB): {}".format(freqs_mean, freqs_std))
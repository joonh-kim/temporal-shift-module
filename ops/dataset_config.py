# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

ROOT_DATASET = '/data1/'  # '/data/jilin/'

def return_somethingv2(modality):
    filename_categories = 'sthsthv2/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'sthsthv2/tsm_img'
        filename_imglist_train = 'sthsthv2/train_videofolder.txt'
        filename_imglist_val = 'sthsthv2/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_minikinetics(modality):
    filename_categories = 200
    if modality in 'RGB':
        root_data = ROOT_DATASET + 'minikinetics/frames/images'
        filename_imglist_train = 'minikinetics/labels/train_videofolder.txt'
        filename_imglist_val = 'minikinetics/labels/val_videofolder.txt'
        prefix = 'image_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'somethingv2': return_somethingv2,
                   'minikinetics': return_minikinetics}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix

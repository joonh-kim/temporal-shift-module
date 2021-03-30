# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py

import os

label_path = '/data1/minikinetics/labels'

if __name__ == '__main__':

    files_input = ['mini_val_videofolder.txt', 'mini_train_videofolder.txt']
    files_output = ['val_videofolder.txt', 'train_videofolder.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(os.path.join(label_path, filename_input)) as f:
            lines = f.readlines()
        videos = []
        frames = []
        labels = []
        for line in lines:
            items = line.split('/')
            res_items = items[3].split(' ')
            videos.append(items[2] + '/' + res_items[0])
            frames.append(res_items[1])
            labels.append(res_items[2])

        output = []
        for i in range(len(videos)):
            curVideo = videos[i]
            curFrames = int(frames[i])
            curLabels = int(labels[i])
            output.append('%s,%d,%d' % (curVideo, curFrames, curLabels))

        with open(os.path.join(label_path, filename_output), 'w') as f:
            f.write('\n'.join(output))

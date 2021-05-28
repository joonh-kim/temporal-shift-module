import numpy as np
from math import pi


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def DCTmatrix(length):
    C = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            if j == 0:
                C[j, i] = np.sqrt(1 / length)
            else:
                C[j, i] = np.sqrt(2 / length) * np.cos((2 * i + 1) * j * pi / (2 * length))
    return C

def DCTmatrix_rgb(C, length):
    C_hat = np.zeros((3 * length, 3 * length))
    for i in range(3 * length):
        if i < length:
            C_hat[i, :length] = C[i, :]
        elif i < 2 * length:
            C_hat[i, length: 2 * length] = C[i - length, :]
        else:
            C_hat[i, 2 * length: 3 * length] = C[i - (2 * length), :]
    return C_hat

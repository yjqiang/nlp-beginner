"""
cs224n a3
"""

from typing import List, Iterator

import numpy as np


def get_minibatches(datas: List[np.ndarray], minibatch_size: int, shuffle: bool = True) -> Iterator[np.ndarray]:
    """
    Iterates through the provided data one minibatch at one time.

    :param datas: a list where each element is numpy array
    :param minibatch_size: the maximum number of items in a minibatch
    :param shuffle: whether to randomize the order of returned data
    :return: returns the next minibatch of each element in the list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.
    """
    data_size = datas[0].shape[0]  # shape 都得是 (N, *) 类型
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start: minibatch_start + minibatch_size]  # indices（打乱或未打乱）中依次切成一段一段，每段长 minibatch_size
        yield [data[minibatch_indices] for data in datas]


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0  # latest value
        self.sum = 0  # summary
        self.count = 0  # 次数
        self.avg = 0  # average

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

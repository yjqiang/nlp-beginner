"""
这是 torch 中使用的 iterator 部分
"""
from typing import List, Iterable, Tuple
import math

import numpy as np
import torch


class MyDataLoader:
    __slots__ = ('num', 'x1', 'x2', 'y', 'batch_size', 'shuffle', 'batch_num')

    def __init__(self, x1: List[torch.Tensor], x2: List[torch.Tensor], y: torch.Tensor, batch_size: int, shuffle: bool = True):
        """
        三种 data x1、x2、y；每种 data 的数量相等（num）
        :param x1: 一组句子，每个句子都是由一个 Tensor 表示，Tensor 不等长（防止填充过大，炸内存）
        :param x2: 一组句子，每个句子都是由一个 Tensor 表示，Tensor 不等长（防止填充过大，炸内存）
        :param y: 一组标签，shape: (N, )
        :param batch_size: batch size
        :param shuffle: whether to randomly shuffle the dataset
        """
        self.num = len(x1)  # 样本总数量
        assert len(x1) == len(x2) == len(y)

        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batch_num = math.ceil(self.num / batch_size)  # 一个 epoch 多少个 iterator

    def __len__(self) -> int:
        return self.batch_num

    def __iter__(self) -> Iterable[Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]]:
        return self.batch_iter()

    def batch_iter(self) -> Iterable[Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]]:
        """ Yield batches of datas.
        """
        index_array = list(range(self.num))

        if self.shuffle:
            np.random.shuffle(index_array)  # 对 index 进行随机化处理

        for start in range(0, self.num, self.batch_size):
            indices = index_array[start: start + self.batch_size]
            yield [self.x1[index] for index in indices], \
                [self.x2[index] for index in indices], \
                torch.tensor([self.y[index] for index in indices])

"""
这是 torch 中使用的 iterator 部分
"""
from typing import Tuple

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, x1: torch.Tensor, x1_seq_lengths: torch.Tensor, x2: torch.Tensor, x2_seq_lengths: torch.Tensor, y: torch.Tensor):
        """
        :param x1: shape: (batch_size, max_sentence_length1)
        :param x2: shape: (batch_size, max_sentence_length2)

        :param x1_seq_lengths: shape: (batch_size,)；每句话的实际长度
        :param x2_seq_lengths: shape: (batch_size,)；每句话的实际长度

        :param y: shape: (batch_size, )   y[i] 是 x[i] 的分类真值，且 0 <= y[i] < CLASS_NUM
        """
        assert x1.shape[0] == len(x1_seq_lengths) == x2.shape[0] == len(x2_seq_lengths) == y.shape[0]
        self.x1 = x1
        self.x2 = x2
        self.x1_seq_lengths = x1_seq_lengths
        self.x2_seq_lengths = x2_seq_lengths
        self.y = y
        self.len = y.shape[0]

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x1[index], self.x1_seq_lengths[index], self.x2[index], self.x2_seq_lengths[index], self.y[index]

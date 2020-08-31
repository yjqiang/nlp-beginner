"""
这是 torch 中使用的 iterator 部分
"""
from typing import Tuple, List

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, seq_lengths: List[int]):
        """
        :param x: shape: (batch_size, max_sentence_length)
        :param y: shape: (batch_size, )   y[i] 是 x[i] 的分类真值，且 0 <= y[i] < CLASS_NUM
        :param seq_lengths: List of actual lengths for each of the sentences in the batch
        """
        assert x.shape[0] == y.shape[0] == len(seq_lengths)
        self.x = x
        self.y = y
        self.seq_lengths = seq_lengths

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.x[index], self.y[index], self.seq_lengths[index]

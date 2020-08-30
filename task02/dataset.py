"""
这是 torch 中使用的 iterator 部分
"""
from typing import Tuple

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        """
        :param x: shape: (batch_size, max_sentence_length)
        :param y: shape: (batch_size, )   y[i] 是 x[i] 的分类真值，且 0 <= y[i] < CLASS_NUM
        """
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

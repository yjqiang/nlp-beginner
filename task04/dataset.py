"""
这是 torch 中使用的 iterator 部分
"""
from typing import Tuple

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, x_sentences_chars: torch.Tensor, x_sentences_words: torch.Tensor, x_sentences_lens: torch.Tensor, y: torch.Tensor):
        """
        :param x_sentences_chars: shape: (N, max_sentence_len, max_word_len) 每个元素都是一个 char 的 index
        :param x_sentences_words: shape: (N, max_sentence_len) 每个元素都是一个 word 的 index
        :param x_sentences_lens: shape: (N,)；每句话的实际长度

        :param y: shape: (N, max_sentence_len)  y[i] 是 x[i][j]（第 i 句的第 j 个 word） 的分类真值
        """
        assert x_sentences_chars.shape[0] == x_sentences_words.shape[0] == len(x_sentences_lens) == y.shape[0]

        self.x_sentences_chars = x_sentences_chars
        self.x_sentences_words = x_sentences_words
        self.x_sentences_lens = x_sentences_lens
        self.y = y

        self.len = y.shape[0]

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_sentences_chars[index], self.x_sentences_words[index], self.x_sentences_lens[index], self.y[index]

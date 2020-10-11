"""
2.1 CNN for Character-level Representation
"""

import torch
from torch import nn

from vocab import CharVocab
from . import parameter_init


class CNN(nn.Module):
    def __init__(self, e_char: int, filter_num: int, window_size: int, padding: int):
        """

        :param e_char: char 的向量维数
        :param filter_num: cnn filter 数目，也是 char cnn 的输出维度（每个 word 的对应向量维数）
        :param window_size: filter 的窗口大小
        :param padding: 左右填充
        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels=e_char, out_channels=filter_num, kernel_size=window_size, padding=padding)
        # max pooling over time 用 MaxPool1d 得输入 kernel_size，老子不爽
        # 用 max 替换了，写在了 forward 里面

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: shape: (N, e_char, max_word_len)  max_word_len 为 word 拥有多少个字母（还有填充等）
        :return: shape: (N, filter_num)
        """
        x_conv = self.conv(x)  # shape: (N, filter_num, max_word_len + 1 - k + padding*2)
        # max pooling over time
        result, _ = torch.max(x_conv, dim=-1)
        return result


class CharCNNEmbedding(nn.Module):
    def __init__(self, e_char: int, filter_num: int, window_size: int, padding: int, dropout_p: float, vocab: CharVocab):
        """
        :param e_char: char 的向量维数
        :param filter_num: cnn filter 数目，也是 char cnn 的输出维度（每个 word 的对应向量维数）
        :param dropout_p: p
        :param window_size: cnn filter 的窗口大小
        :param padding: cnn padding
        :param vocab: CharVocab object. See vocab.py for documentation. 全局共享即可
        """
        super().__init__()

        self.embedding_part = nn.Embedding(num_embeddings=len(vocab.id2char), embedding_dim=e_char, padding_idx=vocab.pad_index)
        self.dropout_part = nn.Dropout(p=dropout_p)
        self.cnn_part = CNN(e_char=e_char, filter_num=filter_num, window_size=window_size, padding=padding)  # (*, e_char, max_word_len=一个单词有多少字母) -> (*, filter_num)

        parameter_init.init_embedding(self.embedding_part)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        :param indices: Tensor of integers of shape (N, max_sentence_len, max_word_len) where each integer is an index into the char vocabulary

        :return: Tensor of shape (N, max_sentence_len, filter_num), containing the CNN-based embeddings for each word of the sentences in the batch
        """
        # (N, max_sentence_len, max_word_len) -> (N, max_sentence_len, max_word_len, e_char)
        x = self.embedding_part.forward(indices)

        # (N, max_sentence_len, max_word_len, e_char) -> (*, e_char, max_word_len)  其中 * = N x max_sentence_len，这里 N 和 max_sentence_len 顺序不要乱（指的是 reshape 这里）
        n, sentence_length, max_word_len, e_char = x.shape
        # shaped_x = x.view(-1, e_char, max_word_len)  # 翻阅笔记你会发现 view/reshape 更像一个对数组 flatten 后（实际是 tensor 底层数据）的一个切割操作，这里用是“错误”的！！
        shaped_x = x.reshape(-1, max_word_len, e_char).transpose(1, 2)

        # 论文中：A dropout layer (Srivastava et al., 2014) is applied before character embeddings are input to CNN.
        shaped_x = self.dropout_part(shaped_x)

        # (*, e_char, max_word_len) -> (*, filter_num)
        x_conv_out = self.cnn_part.forward(shaped_x)
        # (*, filter_num) -> (N, max_sentence_len, filter_num)
        result = x_conv_out.reshape(n, sentence_length, -1)
        return result

from typing import List, Optional

import torch
import torch.nn as nn

from vocab import Vocab


class CNNLayer(nn.Module):
    """
    仅仅实现了一个 CNN 部分
    """
    def __init__(self, embedding_size: int, filter_num: int, window_size: int, max_sentence_length: int, padding: int = 0):
        """
        通过一个本模块后，我们对每个句子（用 N 表示句子数）得到了 filters_nums 个特征
        :param embedding_size: 词向量维度
        :param filter_num: 卷积个数（几个滤波器）
        :param window_size: 窗口大小，对应论文中的 h
        :param padding: 两边填充
        :param max_sentence_length: 每个句子有多少单词（已经根据这个参数进行了填充或截断），对应论文中的 n
        """
        super().__init__()

        # (N, embedding_size, max_sentence_length=论文中的 n) -> (N, filters_nums, n - h + 1 + padding*2)
        self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=filter_num, kernel_size=window_size, padding=padding)
        # 对应论文 f is a non-linear function such as the hyperbolic tangent
        self.relu = nn.ReLU()
        # (N, filters_nums, n - h + 1 + padding*2) -> (N, filters_nums, 1)
        self.max_pool = nn.MaxPool1d(kernel_size=max_sentence_length - window_size + 1 + padding*2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape：(N, embedding_size, max_sentence_length)
        :return:
        """
        # 论文里面的 ci = f(w · x[i]: x[i+h−1] + b)，这里对每句话，有 filters_num *  (n - h + 1 + padding*2) 个 ci
        # 其中论文中的 c = [c1, c2, . . . , cn−h+1]
        c = self.relu(self.conv(x))  # (N, filters_nums, n - h + 1 + padding*2)
        # 论文里面的 ĉ = max{c},这里我们有 filters_nums 个 ĉ（特征）
        c_hat = self.max_pool(c).squeeze(-1)  # (N, filters_nums)
        return c_hat


class CNNModule(nn.Module):
    """
    接收到 N 句话（已经替换了词向量）之后，
    """
    def __init__(self, embedding_size: int, list_filter_nums: List[int], list_window_sizes: List[int], max_sentence_length: int,
                 dropout_p: float, class_num: int, list_paddings: Optional[List[int]] = None):
        """
        参数中带 list_* 的要求 len 相同，即元素个数一致
        :param embedding_size: 词向量维度
        :param list_filter_nums: 卷积个数（几个滤波器）
        :param list_window_sizes: 窗口大小，对应论文中的 h
        :param max_sentence_length: 每个句子有多少单词（已经根据这个参数进行了填充或截断），对应论文中的 n
        :param dropout_p: p – probability of an element to be zeroed.
        :param class_num: 分类数目
        :param list_paddings: 两边填充
        """
        super().__init__()

        if list_paddings is None:
            list_paddings = [0 for _ in list_filter_nums]

        list_cnn_layers = []
        for filter_num, window_size, padding in zip(list_filter_nums, list_window_sizes, list_paddings):
            cnn_layer = CNNLayer(embedding_size=embedding_size, filter_num=filter_num, window_size=window_size, max_sentence_length=max_sentence_length, padding=padding)
            list_cnn_layers.append(cnn_layer)

        self.cnn_layers = nn.ModuleList(list_cnn_layers)

        self.dropout = nn.Dropout(dropout_p)
        self.full_conn = nn.Linear(sum(list_filter_nums), class_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape：(N, max_sentence_length, embedding_size)
        :return:
        """
        x = x.transpose(1, 2)  # (N, max_sentence_length, embedding_size) -> (N, embedding_size, max_sentence_length)

        # 论文中 z = [ĉ1, . . . , ĉm]
        list_z = [cnn_layer(x) for cnn_layer in self.cnn_layers]
        z = torch.cat(list_z, dim=1)  # shape: (N, sum(list_filter_nums))  list_filter_nums 见 init 这里

        # 论文中 y = w · (z ◦ r) + b
        y = self.full_conn(self.dropout(z))  # shape: (N, class_num)  class_num 见 init 这里

        return y


class Model(nn.Module):
    def __init__(self, vocab: Vocab, embedding_size: int, list_filter_nums: List[int], list_window_sizes: List[int], max_sentence_length: int,
                 dropout_p: float, class_num: int, list_paddings: Optional[List[int]] = None, embedding: Optional[nn.Embedding] = None):
        """
        参数中带 list_* 的要求 len 相同，即元素个数一致
        :param vocab: 提取num_embeddings: 词典中的字数； padding_idx: pad
        :param embedding_size: 词向量维度
        :param list_filter_nums: 卷积个数（几个滤波器）
        :param list_window_sizes: 窗口大小，对应论文中的 h
        :param max_sentence_length: 每个句子有多少单词（已经根据这个参数进行了填充或截断），对应论文中的 n
        :param dropout_p: p – probability of an element to be zeroed.
        :param class_num: 分类数目
        :param list_paddings: 两边填充
        :param embedding: 预训练的embedding
        """
        super().__init__()
        if embedding is None:
            num_embeddings = len(vocab)
            padding_idx = vocab.pad_index
            self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_size, padding_idx=padding_idx)
        else:
            self.embedding = embedding
        self.module = CNNModule(embedding_size, list_filter_nums, list_window_sizes, max_sentence_length, dropout_p, class_num, list_paddings)
        self.loss_func = nn.CrossEntropyLoss()

    def get_scores(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: shape: (batch_size, max_sentence_length) batch_size 个句子，每个句子都是 max_sentence_length，每个元素代表对应 word 的 index
        :return:
        """
        embedding_x = self.embedding(x)  # shape: (batch_size, max_sentence_length, EMBEDDING_SIZE)
        scores = self.module.forward(embedding_x)  # shape: (batch_size, CLASS_NUM)
        return scores

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """

        :param x: shape: (batch_size, max_sentence_length) batch_size 个句子，每个句子都是 max_sentence_length，每个元素代表对应 word 的 index
        :param y: shape: (batch_size, )   y[i] 是 x[i] 的分类真值，且 0 <= y[i] < CLASS_NUM
        :return:
        """
        loss = self.loss_func(self.get_scores(x), y)
        return loss

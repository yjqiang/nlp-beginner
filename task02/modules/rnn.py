from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from vocab import Vocab


class Model(nn.Module):
    def __init__(self, vocab: Vocab, embedding_size: int, hidden_size: int, num_layers: int, class_num: int, dropout_p: float, embedding: Optional[nn.Embedding] = None):
        """
        :param vocab: 提取num_embeddings: 词典中的字数； padding_idx: pad
        :param embedding_size: 词向量维度
        :param hidden_size: 隐藏层维数
        :param num_layers: LSTM 层数
        :param class_num: 分类数目
        :param dropout_p: p – probability of an element to be zeroed.
        :param embedding: 预训练的embedding
        """
        super().__init__()
        if embedding is None:
            num_embeddings = len(vocab)
            padding_idx = vocab.pad_index
            self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_size, padding_idx=padding_idx)
        else:
            self.embedding = embedding
        self.module = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_p, bidirectional=True)
        self.full_conn = nn.Linear(2 * hidden_size, class_num)  # 2 for bidirectional
        self.loss_func = nn.CrossEntropyLoss()

    def get_scores(self, x: torch.Tensor, seq_lengths: List[int]) -> torch.Tensor:
        """

        :param x: shape: (batch_size, max_sentence_length) batch_size 个句子，每个句子都是 max_sentence_length，每个元素代表对应 word 的 index
        :param seq_lengths: List of actual lengths for each of the sentences in the batch
        :return:
        """
        embedding_x = self.embedding(x)  # shape: (batch_size, max_sentence_length, EMBEDDING_SIZE)
        packed_x = pack_padded_sequence(embedding_x, lengths=seq_lengths, batch_first=True, enforce_sorted=False)  # 打包准备作为 LSTM 的“输入流”; packed_x 为一个 "a pecked sequence"
        packed_enc_hiddens, (last_hidden, last_cell) = self.module(packed_x)  # packed_enc_hiddens 为 "a pecked sequence"; h_n, c_n 均为 (num_layers * 2, batch_size, hidden_size); 2 for bidirectional
        # 捏合最后一层的 h_n(->) 和 h_1(<-)，使输入变成 (batch_size, 2*hidden_size)，然后全连接层得到 decoder 的初始化 h0
        scores = self.full_conn(torch.cat((last_hidden[-2], last_hidden[-1]), dim=1))  # scores (batch_size, class_num)
        return scores

    def forward(self, x: torch.Tensor, seq_lengths: List[int], y: torch.Tensor) -> torch.Tensor:
        """

        :param x: shape: (batch_size, max_sentence_length) batch_size 个句子，每个句子都是 max_sentence_length，每个元素代表对应 word 的 index
        :param seq_lengths: List of actual lengths for each of the sentences in the batch
        :param y: shape: (batch_size, )   y[i] 是 x[i] 的分类真值，且 0 <= y[i] < CLASS_NUM
        :return:
        """
        loss = self.loss_func(self.get_scores(x, seq_lengths), y)
        return loss

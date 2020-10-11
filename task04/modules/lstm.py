"""
2.2 Bi-directional LSTM
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import parameter_init


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        """ Init NMT Model.

        :param input_size: Figure 3: Then the character representation vector is concatenated with the word embedding before feeding into the BLSTM network.
        :param hidden_size: hidden_size
        :param num_layers: num_layers
        """
        super().__init__()
        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

        parameter_init.init_lstm(self.bi_lstm)

    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        
        :param x: shape: (N, max_sentence_len, input_size)
        :param seq_lengths: shape: (N,) ；每句话的实际长度
        :return: shape: (N, max_sentence_len, 2*hidden_size(因为是 BLSTM))
        """
        max_sentence_len = x.shape[1]
        packed_x = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)  # 函数内自动进行排序
        output, _ = self.bi_lstm(packed_x)
        # pad_packed_sequence(output, batch_first=True) 是根据最长句子进行填充等操作，而不是 pack_padded_sequence 中 input 的最长句长设置进行填充（我们是提前对所有数据集进行了填充，每个 iterator 的最长句长不一定是数据集的最长句子长度）
        # 为了防止这种情况，我们需要 total_length 字段
        result, _ = pad_packed_sequence(output, batch_first=True, total_length=max_sentence_len)
        return result

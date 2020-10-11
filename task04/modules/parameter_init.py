"""
论文 3.1 Parameter Initialization
"""
import math

from torch import nn


def init_embedding(embedding: nn.Embedding) -> None:
    """
    初始化 nn.Embedding 的 ~Embedding.weight；参考论文 3.1 的 Word Embeddings 和 Character Embeddings
    :param embedding:  shape: (num_embeddings, embedding_dim 即向量维数)
    :return:
    """
    x = math.sqrt(3.0 / embedding.weight.shape[1])
    nn.init.uniform_(embedding.weight, -x, x)


def init_linear(linear: nn.Linear) -> None:
    """
    参考论文 3.1 的 Weight Matrices and Bias Vectors
    :param linear:
    :return:
    """
    x = math.sqrt(6.0 / (linear.weight.shape[0] + linear.weight.shape[1]))
    nn.init.uniform_(linear.weight, -x, x)

    linear.bias.data.zero_()  # Bias vectors are initialized to zero


def init_lstm(lstm: nn.LSTM) -> None:
    """
    参考论文 3.1 的 Weight Matrices and Bias Vectors
    :param lstm:
    :return:
    """
    for param in lstm.parameters():
        if param.dim() > 1:  # Weight Matrices
            x = math.sqrt(6 / (param.shape[0] / 4 + param.shape[1]))  # lstm 的 Weight 是 4 个矩阵“粘合”的
            nn.init.uniform_(param, -x, x)
        else:  # Bias Vectors
            # e bias bf for the forget gate in LSTM , which is initialized to 1.0 (Jozefowicz et al., 2015).
            param.data.zero_()
            hidden_size = param.shape[0] // 4
            param.data[hidden_size * 1: hidden_size * 2] = 1  # lstm 的 bias 也是 4 个矩阵“粘合”的

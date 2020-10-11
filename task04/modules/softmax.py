from typing import Tuple

import torch
from torch import nn

from label import Label
from . import parameter_init


class SoftmaxDecoder(nn.Module):
    """
        这部分是负责接受 LSTM 的输出之后，进行求 loss 和 预测；和 crf.py 里面的 CRFDecoder 同样的作用
    """
    def __init__(self, hidden_size: int, label: Label):
        """
        :param hidden_size: (LSTM) 注意由于是 双向 LSTM，所以要除以 2
        :param label: Label object. See label.py for documentation. 全局共享即可
        # 论文中： y0 and yn are the start and end tags of a sentence, that we add to the set of possible tags. A is therefore a square matrix of size k+2
        # 在有效标注之外，我们添加了 START 和 END
        """
        super().__init__()

        self.real_class_num = len(label) - 2

        self.hidden2tag = nn.Linear(hidden_size, self.real_class_num)  # 这里打分不打 START 和 END，没必要你懂吧；这里注意 START 和 END 在 Label 里面 index 要在最后，才能这样用，否则会错误

        parameter_init.init_linear(self.hidden2tag)

    def cal(self, lstm_result: torch.Tensor, x_sentence_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward 和 predict 有很多公共事务，合并
        :param lstm_result: shape: (N, max_sentence_len, hidden_size) LSTM 的结果
        :param x_sentence_lens: 每句话的实际长度；shape: (N,)

        :return: mask  BoolTensor 用于标记每个句子的实际长度，其中 padding 部分为 False；shape: (N, max_sentence_len)
        :return: scores  (N, max_sentence_len, real_class_num(不含 START END)) 每个标签的分数（其实是条件概率）
        """
        logits = self.hidden2tag.forward(lstm_result)  # (N, max_sentence_len, real_class_num(不含 START END))
        scores = torch.nn.functional.softmax(logits, -1)  # (N, max_sentence_len, real_class_num(不含 START END))；条件概率

        n, max_sentence_len, _ = lstm_result.shape
        # torch.arange(max_sentence_len).expand(n, -1) 扩展生成 shape: (n, max_sentence_len)；每一行都是 [0, 1, 2, ...]
        # torch.lt 负责比较(less than 即 <； PS：英文缩写扩展我瞎猜的)
        # mask BoolTensor 用于标记每个句子的实际长度，其中 padding 部分为 False；shape: (N, max_sentence_len)
        mask = torch.lt(torch.arange(max_sentence_len, device=lstm_result.device).expand(n, -1), x_sentence_lens.unsqueeze(-1))
        return scores, mask

    def forward(self, lstm_result: torch.Tensor, x_sentence_lens: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        获取 loss，为了梯度下降
        :param lstm_result: shape: (N, max_sentence_len, hidden_size) LSTM 的结果
        :param x_sentence_lens: 每句话的实际长度；shape: (N,)
        :param y: shape: (N, max_sentence_len)  y[i] 是 x[i][j]（第 i 句的第 j 个 word） 的分类真值  (不含 START END 即 仅含实际句子的所有标签)

        :return: 标量
        """

        scores, mask = self.cal(lstm_result=lstm_result, x_sentence_lens=x_sentence_lens)  # (N, max_sentence_len, real_class_num(不含 START END))
        # cross entropy loss
        # torch.eye 生成 A 2-D tensor with ones on the diagonal and zeros elsewhere；对角线为 1 其余为 0; torch.eye(self.real_class_num) 的 shape: (class_num, class_num)
        # tmp[index: torch.Tensor] 和 nn.Embedding 操作类似，index 的每个元素数值 i 表示 tmp[i] 数据
        # torch.eye(self.real_class_num)[y] 的 shape: (N, max_sentence_len, class_num)
        # one_hot_labels[i, j] 元素表示第 i 句第 j 个 word 的标定 label 情况，是一个列/行向量，label 对应的 index 处为 1 其余为 0
        one_hot_labels = torch.eye(self.real_class_num, device=lstm_result.device)[y]
        # one_hot_labels * scores 挑选出正确的标签所在的 score
        # torch.sum(one_hot_labels * scores, dim=-1) 的 shape: (N, max_sentence_len) ;(one_hot_labels * scores)[i, j] 表示第 i 句第 j 个 word 的标定 label 的 score（条件概率）
        # -log p 就是交叉熵
        losses = -torch.log(torch.sum(one_hot_labels * scores, dim=-1))  # (N, max_sentence_len)
        masked_losses = torch.masked_select(losses, mask)  # 仅计入有效的 loss
        return masked_losses.sum()

    def predict(self, lstm_result: torch.Tensor, x_sentence_lens: torch.Tensor) -> torch.Tensor:
        """
        预测
        :param lstm_result: shape: (N, max_sentence_len, hidden_size) LSTM 的结果
        :param x_sentence_lens: 每句话的实际长度；shape: (N,)

        :return: shape: (N, max_sentence_len)  预测的标注
        """
        scores, mask = self.cal(lstm_result=lstm_result, x_sentence_lens=x_sentence_lens)  # (N, max_sentence_len, real_class_num(不含 START END))
        return torch.argmax(scores, dim=-1)

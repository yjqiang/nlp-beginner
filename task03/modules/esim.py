from typing import Tuple


import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class MLP(nn.Module):
    """
    根据论文 The MLP has a hidden layer with tanh activation and softmax output layer in our experiments.
    这里损失函数写到 model 里面去
    """
    def __init__(self, in_features: int, hidden_features: int, class_num: int):
        """
        :param in_features: 输入的数据维数
        :param hidden_features: 隐藏层的维数
        :param class_num: 分类数目
        """
        super().__init__()
        self.full_conn1 = nn.Linear(in_features, hidden_features)
        self.tanh = nn.Tanh()
        self.full_conn2 = nn.Linear(hidden_features, class_num)
        self.loss_func = nn.CrossEntropyLoss()

    def get_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取分数，为了准确率验证
        :param x: shape: (batch_size, in_features)
        :return: scores shape: (batch_size, class_num)
        """
        return self.full_conn2(self.tanh(self.full_conn1(x)))


class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        """

        :param input_size: 输入维度
        :param hidden_size: h 维数；注意由于 bidirectional 的存在，我们会有 x2 或 /2 操作
        """
        super().__init__()
        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)

    def forward(self, tensor_sentences: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        :param tensor_sentences: (batch_size, max_sentence_length, input_size)
        :param seq_lengths: shape: (batch_size,) ；每句话的实际长度
        :return: shape: (batch_size, max_sentence_length, 2 * hidden_size)  hidden_size 是说 BiLSTM 初始化输入 hidden_size
        """
        packed_sentences = pack_padded_sequence(tensor_sentences, seq_lengths, batch_first=True, enforce_sorted=False)  # 函数内自动进行排序
        output, _ = self.bi_lstm(packed_sentences)
        result, _ = pad_packed_sequence(output, batch_first=True)
        return result


# 论文 3.2
class LocalInferenceModeling(nn.Module):
    """
    https://www.zhihu.com/question/19698133
    y^ (^ 在上) 念 y-hat中文 y-帽
    y- (- 在上) 念 y-bar；中文 y-杠
    y. (.在上) 念 y-dot；中文 y-点
    y~ (~在上) 念 y-tilde；中文 y-波浪
    y→ (→在上) 念 y-arrow；中文 y-箭头
    """
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=2)  # -1

    def attention(self, x1_bar: torch.Tensor, seq_lengths1: torch.Tensor, x2_bar: torch.Tensor, seq_lengths2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        实现 attention，是点积模型
            attention 输入 a_bar 和 b_bar 句子，每个 word 的维度都相同，e_ij = b_bar_j.T @ a_bar_i = a_bar_i.T @ b_bar_j
            1. a_bar 作为查询向量，则 e_ij 看作给定任务相关的查询 a_bar_i 时候，第 j 个输入向量受关注的程度，a_tilde_i = Σ softmax(e_ij) * b_bar_j
            2. b_bar 作为查询向量，则 e_ij 看作给定任务相关的查询 b_bar_j 时候，第 i 个输入向量受关注的程度，b_tilde_j = Σ softmax(e_ij) * a_bar_i
        :param x1_bar: 对应论文中的 a_bar；shape: (batch_size, max_sentence_length1, hidden_size) hidden_size 见 ESIM 的初始化输入
        :param seq_lengths1: 每句话的实际长度；shape: (batch_size,)
        :param x2_bar: 对应论文中的 b_bar；shape: (batch_size, max_sentence_length2, hidden_size) hidden_size 见 ESIM 的初始化输入
        :param seq_lengths2: 每句话的实际长度；shape: (batch_size,)
        :return:
        """
        # x1 shape:(batch_size, max_sentence_length1, hidden_size) ；而 x2.transpose(1, 2) shape:(batch_size, hidden_size, max_sentence_length2)
        # torch.bmm 看作是 batch_size 组矩阵乘法
        e = torch.bmm(x1_bar, x2_bar.transpose(1, 2))  # shape: (batch_size, max_sentence_length1, max_sentence_length2)

        batch_size, max_sentence_length1, max_sentence_length2 = e.shape

        # 各个句子实际长度是不同的，我们把 pad 对应部分屏蔽，让注意力在 pad 部分为 0
        # seq_lengths1.unsqueeze(-1) 生成 shape: (batch_size, 1)
        # torch.arange(max_sentence_length1).expand(batch_size, -1) 扩展生成 shape: (batch_size, max_sentence_length1)；每一行都是 [0, 1, 2, ...]
        # torch.ge 负责比较(greater or equal 即 >=)
        # mask1 为 torch.BoolTensor shape: (batch_size, max_sentence_length1)
        mask1 = torch.ge(torch.arange(max_sentence_length1, device=x1_bar.device).expand(batch_size, -1), seq_lengths1.unsqueeze(-1))
        # mask2 为 torch.BoolTensor shape: (batch_size, max_sentence_length2)
        mask2 = torch.ge(torch.arange(max_sentence_length2, device=x1_bar.device).expand(batch_size, -1), seq_lengths2.unsqueeze(-1))

        # 对 e 进行 masked_fill（使用 mask2）处理之后，由于 mask2.unsqueeze(1) 的存在
        # eg: 第 k 条数据，e 为 （max_sentence_length1, max_sentence_length2)，且 e[:, real_seq_length: max_sentence_length2] 全是 -inf
        # 这样 softmax（softmax(e) 还是 e 的 shape 不变） 之后，softmax(e) @ x2[k] 相当于 softmax(e)[:, real_seq_length] @ [(x2[k]'s 0rd_word), (x2[k]'s 1st_word), (x2[k]'s 2nd_word)...].T
        # 这样 mask2 屏蔽掉了 pad 的 word 部分
        softmax_e = self.softmax(e.masked_fill(mask2.unsqueeze(1), float('-inf')))  # (batch, max_sentence_length1, max_sentence_length2)
        x1_tilde = torch.bmm(softmax_e, x2_bar)  # (batch, max_sentence_length1, hidden_size) 对应 a_tilde
        # mask1 使用同上
        softmax_e = self.softmax(e.transpose(1, 2).masked_fill(mask1.unsqueeze(1), float('-inf')))  # (batch, max_sentence_length2, max_sentence_length1)
        x2_tilde = torch.bmm(softmax_e, x1_bar)  # (batch, max_sentence_length2, hidden_size) 对应 b_tilde
        return x1_tilde, x2_tilde

    @staticmethod
    def enhancement(x_bar: torch.Tensor, x_tilde: torch.Tensor) -> torch.Tensor:
        """
        是为了找到更多特征（差异化？）
        :param x_bar: shape: (batch_size, max_sentence_length, hidden_size) hidden_size 见 ESIM 的初始化输入
        :param x_tilde: 与 x_bar 一样 shape
        :return: 论文中 ma = [a_bar; a_tilde; a_bar − a_tilde; a_bar * a_tilde] mb = [b_bar; b_tilde; b_bar − b_tilde; b_bar * b_tilde]
                 shape: (batch_size, max_sentence_length, hidden_size * 4) hidden_size 见 ESIM 的初始化输入
        """
        return torch.cat([x_bar, x_tilde, x_bar - x_tilde, x_bar * x_tilde], dim=-1)

    def forward(self, x1_bar: torch.Tensor, seq_lengths1: torch.Tensor, x2_bar: torch.Tensor, seq_lengths2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数详细看 attention 和 enhancement
        :param x1_bar:
        :param seq_lengths1:
        :param x2_bar:
        :param seq_lengths2:
        :return: 论文中 ma = [a_bar; a_tilde; a_bar − a_tilde; a_bar * a_tilde] mb = [b_bar; b_tilde; b_bar − b_tilde; b_bar * b_tilde]
                 shape: (batch_size, max_sentence_length, hidden_size * 4) hidden_size 见 ESIM 的初始化输入
        """
        x1_tilde, x2_tilde = self.attention(x1_bar, seq_lengths1, x2_bar, seq_lengths2)
        return self.enhancement(x1_bar, x1_tilde), self.enhancement(x2_bar, x2_tilde)


# 论文 3.3
class InferenceComposition(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, class_num: int):
        """

        :param input_size: 上一层的输出维数；实际赋值时候，input_size = hidden_size * 4，hidden_size 见 ESIM 的初始化输入
        :param hidden_size: hidden_size 就是 ESIM 的初始化输入
        :param class_num: 分类数目
        """
        super().__init__()
        # THE COMPOSITION LAYER
        # We propose to control model complexity in this layer, since the concatenation we described above to compute ma and mb can significantly increase
        # the overall parameter size to potentially overfit the models. We propose to use a mapping F as in Equation (16) and (17).
        # 对应论文中 F，是一个单层神经网络（ReLU作为激活函数），主要用来减少模型的参数避免过拟合
        self.F = nn.Linear(input_size, hidden_size)
        # 论文中 More specifically, we use a 1-layer feedforward neural network with the ReLU activation
        self.relu = nn.ReLU()
        # (batch_size, max_seq_length_i, hidden_size) -> (batch_size, max_seq_length_i, hidden_size)； output 即最后一个 layer 的 h_t
        # (hidden_size // 2) for bidirectional
        self.BiLSTM = BiLSTM(input_size=hidden_size, hidden_size=hidden_size // 2)

        # POOLING
        # (batch_size, max_seq_length_i, hidden_size) -> (batch_size, hidden_size) 即 vi_ave 或 vi_max
        # v = [va_ave; va_max; vb_ave; vb_max].  (batch_size, 4 * hidden_size)

        # A FINAL MULTILAYER PERCEPTRON (MLP) CLASSIFIER
        self.MLP = MLP(in_features=4 * hidden_size, hidden_features=hidden_size, class_num=class_num)
        self.loss_func = self.MLP.loss_func

    def handle_x(self, m_x: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        因为除了最后的 MLP，其他处理过程中，a 和 b 互不影响，且处理步骤一致，所以单独拎出来
        :param m_x: 相当于论文中的 m_a/m_b，shape: (batch_size, max_sentence_length, input_size=hidden_size*4)  hidden_size 见 ESIM 的初始化输入
        :param seq_lengths:  每句话的实际长度；shape: (batch_size,)
        :return: [va_ave; va_max] 或 [vb_ave; vb_max] shape: (batch_size, hidden_size*2)  hidden_size 见 ESIM 的初始化输入
        """
        # THE COMPOSITION LAYER
        v_x_t = self.BiLSTM(self.relu(self.F(m_x)), seq_lengths)  # (batch_size, max_sentence_length, hidden_size)  hidden_size 见 ESIM 的初始化输入；对应论文的 x_a_t/v_b_t

        # POOLING (over-time)
        max_sentence_length = m_x.shape[1]
        v_x_t_transpose = v_x_t.transpose(1, 2)  # (batch_size, hidden_size, max_sentence_length)  方便 over-time pooling
        v_x_avg = F.avg_pool1d(v_x_t_transpose, kernel_size=max_sentence_length).squeeze(-1)  # (batch_size, hidden_size)
        v_x_max = F.max_pool1d(v_x_t_transpose, kernel_size=max_sentence_length).squeeze(-1)  # (batch_size, hidden_size)
        return torch.cat([v_x_avg, v_x_max], dim=1)  # (batch_size, hidden_size*2)  hidden_size 见 ESIM 的初始化输入；对应论文的 x_a_t/v_b_t

    def get_scores(self, m_x1: torch.Tensor, seq_lengths1: torch.Tensor, m_x2: torch.Tensor, seq_lengths2: torch.Tensor) -> torch.Tensor:
        """
        获取分数，为了准确率验证
        :param m_x1: 相当于论文中的 m_a，shape: (batch_size, max_sentence_length1, input_size=hidden_size*4)  hidden_size 见 ESIM 的初始化输入
        :param seq_lengths1: 每句话的实际长度；shape: (batch_size,)
        :param m_x2: 相当于论文中的 m_b，shape: (batch_size, max_sentence_length2, input_size=hidden_size*4)  hidden_size 见 ESIM 的初始化输入
        :param seq_lengths2: 每句话的实际长度；shape: (batch_size,)
        :return: scores shape: (batch_size, class_num)
        """
        # THE COMPOSITION LAYER
        # POOLING (over-time)
        # v = [va_ave; va_max; vb_ave; vb_max]
        v = torch.cat([self.handle_x(m_x1, seq_lengths1), self.handle_x(m_x2, seq_lengths2)], dim=-1)  # shape: (batch_size, 4 * hidden_size)  hidden_size 见 ESIM 的初始化输入

        # A FINAL MULTILAYER PERCEPTRON (MLP) CLASSIFIER
        return self.MLP.get_scores(v)


class ESIM(nn.Module):
    def __init__(self, embedding: nn.Embedding, embedding_size: int, hidden_size: int, class_num: int):
        """
        :param embedding: 预训练的词向量
        :param embedding_size: 词向量维数
        :param hidden_size: h 维数；注意由于 bidirectional 的存在，我们会把 hidden_size // 2！！！！！
        :param class_num: 分类数目
        注意： _i 是因为 premise 和 hypothesis 的填充长度不同罢了； i 取 1 表示 premise（x1；论文中为 a），而 i 取 2 表示 hypothesis（x2；论文中为 b）
        """
        super().__init__()

        assert hidden_size % 2 == 0

        # (batch_size, max_seq_length_i) -> (batch_size, max_seq_length_i, embedding_size)
        self.embedding = embedding

        # 3.1  INPUT ENCODING
        # (batch_size, max_seq_length_i, embedding_size) -> (batch_size, max_seq_length_i, hidden_size)； output 即最后一个 layer 的 h_t
        # (hidden_size // 2) for bidirectional
        # premise 和 hypothesis 共享
        self.BiLSTM = BiLSTM(input_size=embedding_size, hidden_size=hidden_size // 2)

        # 3.2 Local Inference Modeling
        # (batch_size, max_seq_length_i, hidden_size)  -> (batch_size, max_seq_length_i, hidden_size*4)
        self.local_inference_modeling = LocalInferenceModeling()

        # 3.3 INFERENCE COMPOSITION
        # (batch_size, max_seq_length_i, hidden_size * 4) -> loss
        self.inference_composition = InferenceComposition(input_size=hidden_size*4, hidden_size=hidden_size, class_num=class_num)
        self.loss_func = self.inference_composition.loss_func

    def get_scores(self, x1_indices: torch.Tensor, seq_lengths1: torch.Tensor, x2_indices: torch.Tensor, seq_lengths2: torch.Tensor) -> torch.Tensor:
        """
        获取分数，为了准确率验证
        :param x1_indices: premise 句子群；shape: (batch_size, max_sentence_length1)；每个单词对应 vocab 的 index
        :param seq_lengths1: 每句话的实际长度；shape: (batch_size,)
        :param x2_indices: hypothesis 句子群；shape: (batch_size, max_sentence_length2)；每个单词对应 vocab 的 index
        :param seq_lengths2: 每句话的实际长度；shape: (batch_size,)
        :return: scores shape: (batch_size, class_num)
        """
        x1 = self.embedding(x1_indices)  # 论文中的 a；shape: (batch_size, max_sentence_length1, embedding_size)
        x2 = self.embedding(x2_indices)  # 论文中的 b；shape: (batch_size, max_sentence_length2, embedding_size)

        # 3.1  INPUT ENCODING
        x1_bar = self.BiLSTM(x1, seq_lengths1)  # 论文中的 a_bar；shape: (batch_size, max_sentence_length1, hidden_size)
        x2_bar = self.BiLSTM(x2, seq_lengths2)  # 论文中的 b_bar；shape: (batch_size, max_sentence_length2, hidden_size)

        # 3.2 Local Inference Modeling
        m_x1, m_x2 = self.local_inference_modeling.forward(x1_bar, seq_lengths1, x2_bar, seq_lengths2)  # 论文中的 ma/mb；shape: (batch_size, max_sentence_length_i, hidden_size*4)

        # 3.3 INFERENCE COMPOSITION
        scores = self.inference_composition.get_scores(m_x1, seq_lengths1, m_x2, seq_lengths2)  # scores shape: (batch_size, class_num)

        return scores

    def forward(self, x1_indices: torch.Tensor, seq_lengths1: torch.Tensor, x2_indices: torch.Tensor, seq_lengths2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        获取 loss，为了梯度下降
        :param x1_indices: premise 句子群；shape: (batch_size, max_sentence_length1)；每个单词对应 vocab 的 index
        :param seq_lengths1: 每句话的实际长度；shape: (batch_size,)
        :param x2_indices: hypothesis 句子群；shape: (batch_size, max_sentence_length2)；每个单词对应 vocab 的 index
        :param seq_lengths2: 每句话的实际长度；shape: (batch_size,)
        :param y: shape: (batch_size, )   y[i] 是 x[i] 的分类真值，且 0 <= y[i] < CLASS_NUM
        :return: 标量
        """
        scores = self.get_scores(x1_indices, seq_lengths1, x2_indices, seq_lengths2)  # scores shape: (batch_size, class_num)
        return self.loss_func(scores, y)

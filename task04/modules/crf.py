"""
2.3 CRF
本代码文件中的 class_num 包含了 START 和 END
参考了 https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html 细节
"""
from typing import Tuple

import torch
from torch import nn

from label import Label
from . import parameter_init


FLOAT_INF = -1000.0


class CRF(nn.Module):
    def __init__(self, class_num: int, start_tag: int, end_tag: int):
        """
        :param class_num: class_num 即为分类数目；class_num(with START END) = real_class_num + 2
        :param start_tag: class 中 START 对应的 index；我们需要为 START 单独处理（eg：forward_alg）；
                          其他 label（即真实 label） 就像普通 softmax 那样自适应就完事了，但是 START 和 END 我们是要“约束”处理的，必须要与 label.py 对应，否则你用 labels2label_indices 转化正常 label 成了 START 和 END 了
        :param end_tag: class 中 END 对应的 index；我们需要为 END 单独处理（eg：forward_alg）
        # 论文中： y0 and yn are the start and end tags of a sentence, that we add to the set of possible tags. A is therefore a square matrix of size k+2
        # 在有效标注之外，我们添加了 START 和 END
        """
        super().__init__()
        self.class_num = class_num
        self.start_tag = start_tag
        self.end_tag = end_tag
        # 转移矩阵
        self.transition = nn.Parameter(torch.randn(self.class_num, self.class_num))  # transition[j][i] 表示 从 i 跳转到 j 的 score
        self.transition.data[:, self.end_tag] = FLOAT_INF  # 从 END 之后不再跳转
        self.transition.data[self.start_tag, :] = FLOAT_INF  # 不可能跳转到 START

    def _calc_trans_score(self, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        计算转移部分的和
        :param labels: shape: (N, max_sentence_len)  (不含 START END 即 仅含实际句子的所有标签，即为人工标注的真值)
        :param mask: BoolTensor 用于标记每个句子的实际长度，其中 padding 部分为 False；shape: (N, max_sentence_len)
        :return: shape: (N, )
        """
        n, max_sentence_len = labels.size()
        class_num = self.transition.shape[0]  # 包含了 START END 的 class_num

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # block start(与后面 block end 配对): 负责网 labels 人工加入 START END，成为 labels_extend(命名原因：扩展原来的信息)

        # 对于每个句子 {START, [(label_word0, label_word1, ... label_word(sentence_len_i-1)), who_cares, who_cares, ..., who_cares], END}
        # [] 内为原句子，() 内为有效句子
        labels_extend = labels.new_empty((n, max_sentence_len + 2))
        labels_extend[:, 0] = self.start_tag
        labels_extend[:, 1:-1] = labels
        labels_extend[:, -1] = self.end_tag

        pad_one = mask.new_ones([n, 1])  # 全是 1; N 维列向量
        # mask_with_start_end 对于每个句子 {1, [(1, 1, ...), 0, 0, ..., 0], 1}      [] 内为原句子(0 为 padding)；() 内为有效句子
        # long 是因为 gather 要求 index (LongTensor) – the indices of elements to gather
        # shape: (N, max_sentence_len+2)
        mask_with_start_end = torch.cat([pad_one, mask, pad_one], dim=-1).long()
        pad_stop = labels.new_full([n, 1], self.end_tag)  # 全是 END; N 维列向量
        # 我们计算需要的 labels_extend；shape: (N, max_sentence_len + 2)
        # (1 - mask_with_start_end) * pad_stop 得到了 {0, [(0, 0, ...), END, END, ..., END], 0}
        # mask_with_start_end * labels_extend 得到了 {START, [(label_word0, label_word1, ... label_word(sentence_len_i-1)), 0, 0, ..., 0],END}
        # 对于每个句子 {START, [(label_word0, label_word1, ... label_word(sentence_len_i-1)), END, END, ... , END],END}
        labels_extend = (1 - mask_with_start_end) * pad_stop + mask_with_start_end * labels_extend

        # block end: 任务完成
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        transition_expanded = self.transition.unsqueeze(dim=0).expand(n, -1, -1)  # shape: (N, class_num, class_num)
        labels_r = labels_extend[:, 1:]  # shape: (N, max_sentence_len + 1)
        # shape: (N, max_sentence_len + 1, class_num)
        # labels_r_expanded[i, : , any] 第 i 个句子，{[(label_word0, label_word1, ... label_word(sentence_len_i-1)), END, END, ..., END], END}  第一个虚拟 label（即 START） 去掉
        # labels_r_expanded[i, j , any] = labels_r[i, j] = labels_extend[i, j + 1]
        # any 表示 二维矩阵 labels_r_expanded[i] 每行的 value 都相等（即所有列都是同一个列向量）
        labels_r_expanded = labels_r.unsqueeze(-1).expand(-1, -1, class_num)
        # transition_row[i, j, k] = transition_expanded[i, labels_r_expanded[i, j, k], k]
        transition_row = torch.gather(transition_expanded, dim=1, index=labels_r_expanded)

        # shape: ((N, max_sentence_len + 1, 1)
        # labels_l_expanded[i, : , 0] 第 i 个句子，{START, [(label_word0, label_word1, ... label_word(sentence_len_i-1)), END, END, ... ]}  最后一个虚拟 label（即 END） 去掉
        # labels_l_expanded[i, j , 0] = labels_extend[i, j]
        labels_l_expanded = labels_extend[:, :-1].unsqueeze(-1)
        # shape: (N, max_sentence_len + 1, 1)
        # transition_matrix[i, j, 0] = transition_row[i, j, labels_l_expanded[i, j, 0]] = transition_expanded[i, labels_r_expanded[i, j, labels_l_expanded[i, j, 0]], labels_l_expanded[i, j, 0]]
        # = transition_expanded[i, labels_r_expanded[i, j, any],  labels_l_expanded[i, j, 0]] = transition_expanded[i, labels_extend[i, j+1],  labels_extend[i, j]]
        # transition_expanded[i, labels_extend[i, j+1],  labels_extend[i, j]] 就是 对于第 i 句话，从 A[y_(i),y_(i+1)] 的转移情况
        transition_matrix = torch.gather(transition_row, dim=2, index=labels_l_expanded)
        # shape: (N, max_sentence_len + 1)
        # transition_matrix[i, j] = transition_expanded[i, labels_extend[i, j+1],  labels_extend[i, j]] = self.transition[labels_extend[i, j+1],  labels_extend[i, j]] 表示
        # 第 i 句从第 j 个 word 的标签转移到第 j+1 个标签的转移分数
        # labels_extend[i, 0] 是 START，labels_extend[i, max_sentence_len] 是最后一个原句子的单词(可能是 padding)的 label(padding 对应了 END)， labels_extend[i, max_sentence_len+1] 是 END
        # transition_matrix[i, k=0] 为 START -> label_word0；k 比较小 transition_matrix[i, k] 为 label_word(k-1) -> label_word(k)；transition_matrix[i, k=sentence_len_i] 为 label_word(sentence_len_i-1) -> END
        # k>sentence_len_i 时 transition_matrix[i, k=0] 为 END -> END
        # 大致来说 labels_extend[i, :] 就是  START -> [(label_word0 -> label_word1 -> ... label_word(sentence_len_i-1)) -> END -> END, ...] -> END.
        transition_matrix = transition_matrix.squeeze(-1)

        # START -> [(label_word0 -> label_word1 -> ... label_word(sentence_len_i-1)) -> END -> END, ...] -> END
        # 有效序列为 START -> label_word0 -> label_word1 -> ... label_word(sentence_len_i-1) -> END；长度为 sentence_len_i + 1，所以每个句子的 mask 需要增 1
        # shape: (N, max_sentence_len + 1)
        mask_with_start = torch.cat([pad_one, mask], dim=-1).float()
        transition_matrix = transition_matrix * mask_with_start

        return torch.sum(transition_matrix, dim=1)

    @staticmethod
    def _calc_emission_score(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        计算发射部分的和
        :param logits: logits typically become an input to the softmax function（eg：tf.nn.softmax）；
                       shape: (N, max_sentence_len, class_num(其实含不含 START END 结果都一样，因为 P[虚拟头, START] = P[虚拟尾, END] = 0)，代码中我们含了) FloatTensor class_num 即为分类数目；是 Emission 矩阵
        :param labels: shape: (N, max_sentence_len)  (不含 START END 即 仅含实际句子的所有标签)
        :param mask: BoolTensor 用于标记每个句子的实际长度，其中 padding 部分为 False；shape: (N, max_sentence_len)

        :return: shape: (N, )
        """
        # shape: (N, max_sentence_len, 1)
        labels_unsqueeze = labels.unsqueeze(-1)
        # 设 torch.gather(logits, dim=2, index=labels_unsqueeze) 得到 emission_matrix'；emission_matrix'[i, j, 0] = logits[i, j, labels_unsqueeze[i, j, 0]] = logits[i, j, labels[i, j]]
        # squeeze 之后 emission_matrix[i, j] = logits[i, j, labels[i, j]] 即表示第 i 句子，第 j 个单词发射矩阵元素 P[j, y_j]
        # shape: (N, max_sentence_len)
        emission_matrix = torch.gather(logits, dim=2, index=labels_unsqueeze).squeeze(-1)
        emission_matrix = emission_matrix * mask.float()
        return torch.sum(emission_matrix, dim=1)

    def get_sentence_score(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        根据某个句子的给定 labels 求 s(y, x) = ΣA[y_(i),y_(i+1)] + ΣP[i, y_(i)]
        :param logits: logits typically become an input to the softmax function（eg：tf.nn.softmax）；
                       shape: (N, max_sentence_len, class_num(其实含不含 START END 结果都一样，因为 P[虚拟头, START] = P[虚拟尾, END] = 0)) FloatTensor class_num 即为分类数目；是 Emission 矩阵
        :param labels: shape: (N, max_sentence_len)  (不含 START END 即 仅含实际句子的所有标签)
        :param mask: BoolTensor 用于标记每个句子的实际长度，其中 padding 部分为 False；shape: (N, max_sentence_len)

        :return: 论文中的 s(X, y) shape: (N,)
        """
        trans_score = self._calc_trans_score(labels=labels, mask=mask)
        emission_score = self._calc_emission_score(logits=logits, labels=labels, mask=mask)
        return emission_score + trans_score

    def forward_alg(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Total score of all sequences. 即 log[(Σy')exp(s(y', x))]  s 就是 s(y, x) = ΣA[y_(i),y_(i+1)] + ΣP[i, y_(i)]
        :param logits: logits typically become an input to the softmax function（eg：tf.nn.softmax）；
                       shape: (N, max_sentence_len, class_num(含 START END)) FloatTensor class_num 即为分类数目；是 Emission 矩阵
        :param mask: BoolTensor 用于标记每个句子的实际长度，其中 padding 部分为 False；shape: (N, max_sentence_len)

        :return: shape: (N,)
        """
        n, max_sentence_len, class_num = logits.shape  # class_num(含 START END)

        # 限制第 - 1 个单词为 START 的 “所有路径” 的 （截至目前的）累计 score 为 0，但是其他 label 的 score 为 -inf
        # 这样下面循环中，对非 START 起步的路径 exp^(路径) 逼近 0，那就基本不考虑了
        # shape: (N, class_num(含 START END))
        log_sum_of_exp = logits.new_full((n, class_num), FLOAT_INF)  # 全 fill_value 的矩阵，使用 logits 的 new_full 是方便直接把 pre_vit 的 device 和 dtype 弄成和 logits 一致；class_num 有 START 和 END
        log_sum_of_exp[:, self.start_tag] = 0

        mask_t = mask.transpose(1, 0)  # shape: (max_sentence_len, N)
        logits_t = logits.transpose(1, 0)  # shape: (max_sentence_len, N, class_num)

        # 每批次（每个 index），处理所有句子的第 index 个单词
        # log_sum_of_exp[k, i] 表示第 k 句话，截至目前为止并限制第 index (for 循环 for index, ... in enumerate(mask_t, logits_t), 无虚拟头尾)个单词为 label_i 的所有前缀路径长度 log(e^路径 0 分数 + e^路径 1 分数 + ...)
        for cur_mask_t, cur_logit_t in zip(mask_t, logits_t):  # cur_mask_t 为 (N,), cur_logit_t 为 (N, class_num(含 START END))
            # (N, class_num(含 START END)) -加维度-> (N, 1, class_num(with START END)) -"复制"-> (N, class_num(含 START END), class_num(含 START END))
            # 在 log_sum_of_exp_expanded[k] 矩阵，每一列上所有 value 均相等（即所有行都是同一个行向量）
            # log_sum_of_exp_expanded[k, *, i] = log_sum_of_exp[k, i] 表示第 k 句话，限制第 index-1 个单词为 label_i 的 “所有截至目前为止路径” 的 log(e^路径 0 分数 + e^路径 1 分数 + ...)
            log_sum_of_exp_expanded = log_sum_of_exp.unsqueeze(dim=1).expand(-1, class_num, -1)

            # (class_num(含 START END), class_num(含 START END)) -加维度-> (1, class_num(含 START END), class_num(含 START END)) -"复制"-> (N, class_num(含 START END), class_num(含 START END))
            # transition_expanded[*, j, i] 表示 从 i 跳转到 j 的 score
            transition_expanded = self.transition.unsqueeze(dim=0).expand(n, -1, -1)

            # (N, class_num(含 START END)) -加维度-> (N, class_num(含 START END), 1) -"复制"-> (N, class_num(含 START END), class_num(含 START END))
            # 对每个句子，每一行上所有 value 均相等（即所有列都是同一个列向量）；logit_expanded[k, j, *] = cur_logit_t[k, j] 表示第 k 句话，限制第 index 个单词为 label_j 的“发射分数”（即 P_(index, yj)）
            logit_expanded = cur_logit_t.unsqueeze(dim=-1).expand(-1, -1, class_num)

            # sum_result[k, j, *] 表示第 k 句话，截至并限制第 index 个单词为 label_j 的 “所有路径”的“和”，具体分析如下
            # sum_result[k, j, t] = log_sum_of_exp_expanded[k, j, t] + transition_expanded[k, j, t] + logit_expanded[k, j, t]
            # = 上一个循环即第 index-1 个的 log_sum_of_exp[k, t] + self.transition[j, t] + logit_t[index, k, j]
            # = 第 k 句， log(e^截至第 index-1 个单词且以 label_t 为结尾的路径 0 分数  + e^截至第 index-1 个单词且以 label_t 为结尾的路径 1 分数 + ...) + trans[label_j, label_t] + 第 index 个单词选取 label_j 的 emission 打分
            # 我们对上式子先 exp 操作，可得到下式子
            # e^[(截至第 index-1 个单词且以 label_t 为结尾的路径 0 分数 + trans[label_j, label_t] + 第 index 个单词选取 label_j 的打分)
            # + e^(截至第 index-1 个单词且以 label_t 为结尾的路径 1 分数 + trans[label_j, label_t] + 第 index 个单词选取 label_j 的打分) + ...]
            # 即 exp(sum_result[k, j, t]) = 第 k 句，截至第 index 个单词 且 第 index-1 个单词选取 label_t 且 第 index 个单词选取 label_j 的 “所有路径”的“exp 和”
            # 那么 sum（dim=2）之后，sum(exp(sum_result[k, j, t]), dim=2)[k, j] 表示第 k 句，截至第 index 个单词 且 第 index 个单词选取 label_j 的 “所有路径”的“exp 和”
            sum_result = log_sum_of_exp_expanded + transition_expanded + logit_expanded
            # log_sum_of_exp_next[k, j] 表示第 k 句话，截至并限制第 index 个单词为 label_j 的 “所有路径”的 log(e^路径 0 分数  + e^路径 1 分数  + ...)
            # shape: (N, class_num(含 START END))
            log_sum_of_exp_next = torch.logsumexp(sum_result, dim=2)

            # 检查有效单词情况
            # 所有句子的第 index 单词是否为有效单词，无效（即某一句长度很短）的时候 log_sum_of_exp_next[k, j] 不应该更新到下次迭代的 log_sum_of_exp 中，我们就到句子真实句子的最后一个有效单词即可
            # (N,) -加维度-> (N, 1) -"复制"-> (N, class_num(含 START END))
            # 若第 k 句话在 index 处为 padding，则 cur_mask[k, :] = cur_mask_t[k] = mask_t[index, k] = mask[k, index] 为 0；否则为 1
            cur_mask = cur_mask_t.float().unsqueeze(dim=-1).expand(n, class_num)
            # 第 k 句话当前长度处的单词为有效单词，log_sum_of_exp_next[k, :] 有效，否则 log_sum_of_exp[k, :] 不变
            log_sum_of_exp = cur_mask * log_sum_of_exp_next + (1 - cur_mask) * log_sum_of_exp

        # 令 MATRIX_END=self.transition[self.end_tag].unsqueeze(0).expand(n, class_num)  shape 为 (N, class_num(含 START END))；该矩阵的每一列都相等（即所有行都是同一个行向量），MATRIX_END[*, j] 表示 第 j 个标签转移到 END 的 score
        # MATRIX_END[*, j] = self.transition[self.end_tag, label_j]
        # 而且 P[任意句子的最后一个单词（虚拟单词）, END]=0
        # 对于那些填充了 padding 的句子，cur_mask 使得 log_sum_of_exp[i, :] 矩阵在超出真正长度之后不再改变，所以在最后/在真正有效最后 word 处 加上 END 的 log_sum_of_exp[i, :] 都一样
        # log_sum_of_exp[k, j] + MATRIX_END[k, j] = 第 k 句话 log(e^结尾为 label_j 的路径0 + e^结尾为 label_j 的路径1 + ...) + trans[END, label_j]  + 0
        # 按照列进行 logsumexp 之后，变成 log(e^路径0(结尾为 END) + e^路径1(结尾为 END) + ...)
        sum_result = log_sum_of_exp + self.transition[self.end_tag].unsqueeze(dim=0).expand(n, class_num)
        # shape: (N,)
        total_score = torch.logsumexp(sum_result, dim=1)
        return total_score

    def viterbi_decode(self, logits: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        动态规划求最优解（有点类似数字三角形问题）；预测
        :param logits: logits typically become an input to the softmax function（eg：tf.nn.softmax）；
                       shape: (N, max_sentence_len, class_num(含 START END)) FloatTensor class_num 即为分类数目；即 Emission 矩阵
        :param mask: BoolTensor 用于标记每个句子的实际长度，其中 padding 部分为 False；shape: (N, max_sentence_len)

        :return best_scores: shape: (N,) FloatTensor
        :return best_paths: shape: (N, max_sentence_len)  预测的标注
        """
        n, max_sentence_len, class_num = logits.shape

        # 初始化后 vit[k, i] 表示第 k 句话，截至目前为止并限制第 index 个(见 for 循环)单词为 label_i 的最大前缀路径长度；限制第 - 1 个单词为 START 的 “最优路径候选者” 的 （截至目前的）累计 score 为 0，但是其他 label 的 score 为 -inf
        # 这样下面循环中，对第 0 个单词的迭代时候的 vit_argmax 一定是 START
        vit = logits.new_full((n, class_num), FLOAT_INF)  # 全 fill_value 的矩阵，使用 logits 的 new_full 是方便直接把 vit 的 device 和 dtype 弄成和 logits 一致
        vit[:, self.start_tag] = 0

        mask_t = mask.transpose(1, 0)  # shape: (max_sentence_len, N)
        logits_t = logits.transpose(1, 0)  # shape: (max_sentence_len, N, class_num)
        pointers = []

        # 每批次（即每个 index），处理所有句子的第 index 个单词
        for index, logit in enumerate(logits_t):  # logit shape: (N, class_num(含 START END)) 表示第 index 个单词的发射矩阵分数状况
            # (N, class_num(含 START END)) -加维度-> (N, 1, class_num(含 START END)) -"复制"-> (N, class_num(含 START END), class_num(含 START END))
            # pre_vit_expanded[k, *, i] = vit[k, i] 表示第 k 句话，截至第 index - 1 个单词为止且限制第 index - 1 个单词为 label_i（只限制“末尾”）的 “最优路径候选者” 的 （截至目前的）累计 score
            # 第一个循环时，截至第 - 1 个单词为止且第 -1 个单词为 START 的累计 score 为 0，第 -1 个单词的 label 为其他的累计 score 很低
            # 在第 k 个矩阵，每一列上所有 value 均相等（即所有行都是同一个行向量）
            pre_vit_expanded = vit.unsqueeze(dim=1).expand(-1, class_num, -1)
            # (class_num(含 START END), class_num(含 START END)) -加维度-> (1, class_num(含 START END), class_num(含 START END)) -"复制"-> (N, class_num(含 START END), class_num(含 START END))
            # transition_expanded[*, j, i] = self.transition[j, i] 表示 从 i 跳转到 j 的 score
            transition_expanded = self.transition.unsqueeze(dim=0).expand(n, -1, -1)

            # vit_trn_sum[k, j, i] = pre_vit_expanded[k, j, i] + transition_expanded[k, j, i] = vit[k, i] + self.transition[j, i]
            # vit_trn_sum[k, j, i] 表示第 k 句话，第 index - 1 个单词为 label_i 的 “最优路径候选者”，从 i 处跳至 j 处后的 score
            # 即在第 k 个矩阵，第 i 列分别表示，从上一层状态(label_i)跳至 0、1、2...
            # 在第 k 个矩阵，第 j 行分别表示，从上一层状态(label_0、1、2...)跳至 label_j
            # 动态规划推导，限制第 index 个单词为 label_j 的 “最优路径候选者”就是从上一层的所有不同 label 结尾的候选者 score + 本轮次跳到 j 的 score 选出的，即每行选出最大的就是 限制第 index 个单词为 label_j 的 “最优路径候选者” 的累计 score
            # 注意！ 这里每一个比较组（即 max{..}）里面在本层都是以同一个 label 为结尾的，所以 P[i, label_j] 可以比较之后，再加上即可
            # shape: (N, class_num(含 START END), class_num(含 START END))
            vit_trn_sum = pre_vit_expanded + transition_expanded
            # vit_max 为 最大数值，vit_argmax 为 vit_max 对应的 index；两者 shape 均为 (N, class_num(含 START END))
            # dim=2 结果消除了第 2 维度，即 vit_trn_sum[k] 每行仅一个保留了下来
            # vit_max[*, j] 就是限制第 index 个单词为 label_j 的 “最优路径候选者”，它的累计 score（不含 P[index, label_j]）
            # vit_argmax[*, j] 是限制第 index 个单词为 label_j 的 “最优路径候选者”，它的第 index-1 个单词的 label
            # 超出句子实际长度无所谓，后面我们会处理（不同句子单独寻路） 超出句子实际的继续执行“转移”(即 vit_argmax 仍有值)，但是 vit[i, :] 矩阵不再改变
            vit_max, vit_argmax = vit_trn_sum.max(dim=2)
            # logit 表示每个句子都各自的 P[i, label_j] 的 score
            # 正如前面提到的，先比较除 P 之外的，再把结果加上 P；
            # 即 max(pre[上一层以 label0 为结尾的候选者] + transition[j, 0], pre[上一层以 label1 为结尾的候选者] + transition[j, 1]...) + P[index, j]
            # 这样的效果与 max(pre[上一层以 label0 为结尾的候选者] + transition[j, 0] + P[index, j], pre[上一层以 label1 为结尾的候选者] + transition[j, 1] + P[index, j]...) 效果一样的
            vit_next = vit_max + logit

            pointers.append(vit_argmax)

            # 检查有效单词情况
            # 所有句子的第 index 单词是否为有效，无效（即某一句长度很短）的时候 cur_mask[k, j] 不应该更新
            # (N,) -加维度-> (N, 1) -"复制"-> (N, class_num(含 START END))
            # 若第 k 句话在 index 处为 padding，则 cur_mask[k, :] = mask_t[index, k] = mask[k, index] 为 0；否则为 1
            cur_mask = mask_t[index].float().unsqueeze(dim=-1).expand(n, class_num)
            # 某句子当前长度处的单词为有效单词，vit_nxt 有效，否则不变
            vit = cur_mask * vit_next + (1 - cur_mask) * vit

        # self.transition[self.end_tag].unsqueeze(0).expand(n, class_num(含 START END))  shape 为 (N, class_num(含 START END))；该矩阵的每一行都相等，matrix[*, j] 表示 第 j 个标签转移到 END 的 score
        # 而且 P[最后一个单词（虚拟单词）, END]=0
        # 对于那些填充了 padding 的句子，vit[i, :] 矩阵不再改变，在最后/在真正有效最后 word 处 加上 END 的 vit[i, :] 都一样
        vit += self.transition[self.end_tag].unsqueeze(dim=0).expand(n, class_num)

        # pointers (N, max_sentence_len, class_num(含 START END))
        # 对每个句子，我们用 pointers[k] (shape: max_sentence_len, class_num(含 START END)) 矩阵表示每个路线情况
        # 假设 START->a->b->c->d->END 为最优路径，则M[-1, d]=c M[-2, c]=b
        # M[0, :] 一定都是 START
        pointers = torch.stack(pointers, dim=1)

        # 最佳 best_score shape: (N,)
        # 最佳 best_labels shape: (N,) 表示最佳路径中最后一个有效单词的 label （vit = cur_mask * vit_next + (1 - cur_mask) * vit 使得pre_vit 的 padding 处与最后一个有效单词处数据一样）
        best_scores, best_labels = vit.max(dim=1)

        # 反向回溯(注意不同句子的路长不同，不同句子单独寻路)
        # 感谢 https://zhuanlan.zhihu.com/p/97858739
        best_paths = []
        for i in range(n):
            # 当前句子的实际长度
            seq_len_i = int(mask[i].sum())
            # 截断 shape: (seq_len_i, class_num(含 START END))
            # pointers_i[index][t] 表示若该句话最佳路径中倒数第 index 个有效 word（从 0 开始计数）的选择 label_t, “下一步”有效 word 的选择 label
            pointers_i = pointers[i, :seq_len_i]

            # 最优路径回溯算法中上一步的 label
            pre_best_label_i = best_labels[i]
            best_path_i = [pre_best_label_i]
            for index in range(seq_len_i-1, -1, -1):  # 从后向前遍历
                # 当前时刻的best_label_i
                cur_best_label_i = pointers_i[index, pre_best_label_i]
                best_path_i.append(cur_best_label_i)
                pre_best_label_i = cur_best_label_i

            # 我们回溯是反向的（倒着从最后一个单词回溯的）
            best_path_i.reverse()
            # best_path_i[1:] 去掉 START
            best_path_i = best_path_i[1:] + [0] * (max_sentence_len - seq_len_i)
            # 添加到总路径中
            best_paths.append(best_path_i)

        best_paths = torch.tensor(best_paths, device=logits.device)  # shape: (N, max_sentence_len)
        return best_scores, best_paths


class CRFDecoder(nn.Module):
    """
    这部分是负责接受 LSTM 的输出之后，进行求 loss 和 预测；和 softmax.py 里面的 SoftmaxDecoder 同样的作用
    """
    def __init__(self, hidden_size: int, label: Label):
        """
        :param hidden_size: (LSTM) 注意由于是 双向 LSTM，所以要除以 2
        :param label: Label object. See label.py for documentation. 全局共享即可
        # 论文中： y0 and yn are the start and end tags of a sentence, that we add to the set of possible tags. A is therefore a square matrix of size k+2
        # 在有效标注之外，我们添加了 START 和 END
        """
        super().__init__()
        # 生成 P （发射矩阵 emission）
        self.hidden2tag = nn.Linear(hidden_size, len(label))
        self.crf_part = CRF(class_num=len(label), start_tag=label.start_index, end_tag=label.end_index)

        parameter_init.init_linear(self.hidden2tag)

    def cal(self, lstm_result: torch.Tensor, x_sentence_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward 和 predict 有很多公共事务，合并
        :param lstm_result: shape: (N, max_sentence_len, hidden_size) LSTM 的结果
        :param x_sentence_lens: 每句话的实际长度；shape: (N,)

        :return: mask BoolTensor 用于标记每个句子的实际长度，其中 padding 部分为 False；shape: (N, max_sentence_len)
            emission 发射矩阵 shape: (N, max_sentence_len, class_num)  包含了 START 和 END；发射矩阵并没有为每个句子进行额外增加虚拟头和尾部，只是打分时候对每个单词额外打了 START 和 END
        """
        emission = self.hidden2tag.forward(lstm_result)  # shape: (N, max_sentence_len, class_num)  包含了 START 和 END

        n, max_sentence_len, _ = emission.shape
        # torch.arange(max_sentence_len).expand(n, -1) 扩展生成 shape: (n, max_sentence_len)；每一行都是 [0, 1, 2, ...]
        # torch.lt 负责比较(less than 即 <； PS：英文缩写扩展我瞎猜的)
        # mask BoolTensor 用于标记每个句子的实际长度，其中 padding 部分为 False；shape: (N, max_sentence_len)
        mask = torch.lt(torch.arange(max_sentence_len, device=lstm_result.device).expand(n, -1), x_sentence_lens.unsqueeze(-1))

        return emission, mask

    def forward(self, lstm_result: torch.Tensor, x_sentence_lens: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        获取 loss，为了梯度下降
        :param lstm_result: shape: (N, max_sentence_len, hidden_size) LSTM 的结果
        :param x_sentence_lens: 每句话的实际长度；shape: (N,)
        :param y: shape: (N, max_sentence_len)  y[i] 是 x[i][j]（第 i 句的第 j 个 word） 的分类真值  (不含 START END 即 仅含实际句子的所有标签)

        :return: 标量
        """

        emission, mask = self.cal(lstm_result=lstm_result, x_sentence_lens=x_sentence_lens)

        # shape: (N,)
        # -log[p(y|x)] = -log[exp(s(y, x)) / (Σy')exp(s(y', x))] = log[(Σy')exp(s(y', x))] - s(y, x)
        losses = self.crf_part.forward_alg(logits=emission, mask=mask) - self.crf_part.get_sentence_score(logits=emission, labels=y, mask=mask)

        return losses.mean()

    def predict(self, lstm_result: torch.Tensor, x_sentence_lens: torch.Tensor) -> torch.Tensor:
        """
        预测
        :param lstm_result: shape: (N, max_sentence_len, hidden_size) LSTM 的结果
        :param x_sentence_lens: 每句话的实际长度；shape: (N,)

        :return: shape: (N, max_sentence_len)  预测的标注
        """
        emission, mask = self.cal(lstm_result=lstm_result, x_sentence_lens=x_sentence_lens)

        _, best_paths = self.crf_part.viterbi_decode(logits=emission, mask=mask)

        return best_paths

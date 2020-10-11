"""
负责把 char 转为 index
"""

from typing import List, Optional

import torch


class Label:
    """ Label Entry.
    """

    def __init__(self):
        bioes_tags = ['B', 'I', 'O', 'E', 'S']
        entities = ['PER', 'ORG', 'LOC', 'MISC']  # 人名、组织名、地名、其他

        labels = ['O'] + [f'{bioes_tag}-{entity}' for bioes_tag in bioes_tags if bioes_tag != 'O' for entity in entities] + ['START', 'END']  # 分类(含 START END)

        self.label2id = dict()

        # self.label2id['PAD'] = 0  # <pad> 额外的 pad 没卵用，还得额外训练参数!!!!!随便挑一个就行（本代码用了 0）
        #
        # self.pad_index = self.label2id['PAD']

        for label in labels:
            self.label2id[label] = len(self.label2id)
        # reverse label2id
        self.id2label = {index: label for label, index in self.label2id.items()}

        self.start_index = self.label2id['START']
        self.end_index = self.label2id['END']
        self.pad_index = 0

    def __len__(self) -> int:
        return len(self.label2id)

    def labels2label_indices(self, labels: List[List[str]]) -> List[List[int]]:
        """ Convert list of sentences of words' labels into list of list of list of label indices.
        :param labels: 每个 str 都是一个 label，对应一个 word; List[str] 就是一句话的所有 label
        :return label_ids (List[List[int]]): 对所有 label 进行替换，注意 labels 我们已经穷举完毕了（即不存在未知的 label）
        """
        return [[self.label2id[word_label] for word_label in sentence_label] for sentence_label in labels]

    def label_indices2labels(self, label_indices: List[List[int]]) -> List[List[str]]:
        """ labels2label_indices 的反向操作
        :param label_indices: 每个 int 都是一个 label，对应一个 word; List[int] 就是一句话的所有 label
        :return label_ids (List[List[int]]): 对所有 label 进行替换，注意 labels 我们已经穷举完毕了（即不存在未知的 label）
        """
        return [[self.id2label[word_label_index] for word_label_index in sentence_label_indices] for sentence_label_indices in label_indices]

    def to_tensor(self, labels: List[List[int]], device: Optional[torch.device]) -> torch.Tensor:
        """ Convert list of sentences (labels of words) into tensor with necessary padding for shorter sentences.

        :param labels: int 为一个 word 的 label index(使用 vocab 的 label2id 处理了)，List[int] 为一个句子的所有 label；
        :param device: device on which to load the tensor, i.e. CPU or GPU

        :return tensor_labels_padded: tensor of (N, max_sentence_len) torch.long 是因为 gather 的 indices 输入必须为 long
        """

        tensor_labels_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(sentence_labels, dtype=torch.long, device=device) for sentence_labels in labels],
                                                               batch_first=True,
                                                               padding_value=self.pad_index)
        return tensor_labels_padded

    @staticmethod
    def to_list(x_sentence_lens: torch.Tensor, y: torch.Tensor) -> List[List[int]]:
        """to_tensor 的反向操作
        :param x_sentence_lens: 每句话的实际长度；shape: (N,)
        :param y: shape: (N, max_sentence_len)  y[i] 是 x[i][j]（第 i 句的第 j 个 word） 的分类真值  (不含 START END 即 仅含实际句子的所有标签)
        :return:
        """

        n = x_sentence_lens.shape[0]
        result = [y[i][:x_sentence_lens[i]].tolist() for i in range(n)]  # 仅截取有效部分
        return result

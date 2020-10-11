"""
负责把 char 转为 index
"""

from typing import List, Optional, Dict

import torch

from utils import utils


class CharVocab:
    """ Vocabulary Entry, i.e. structure containing either
    src or tgt language terms.
    """

    def __init__(self):

        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]"

        self.char2id = dict()  # Converts characters to integers
        self.char2id['∏'] = 0  # <pad> token
        self.char2id['{'] = 1  # start of word token
        self.char2id['}'] = 2  # end of word token
        self.char2id['Û'] = 3  # <unk> token

        self.pad_index = self.char2id['∏']
        self.unk_index = self.char2id['Û']
        self.start_index = self.char2id["{"]
        self.end_index = self.char2id["}"]

        for char in chars:
            self.char2id[char] = len(self.char2id)
        # reverse char2id
        self.id2char = {index: char for char, index in self.char2id.items()}

    def words2char_indices(self, sentences: List[List[str]]) -> List[List[List[int]]]:
        """ Convert list of sentences of words into list of list of list of character indices.
        把每个 word 加上 Start Token 和 End Token 后，转为 List[int]
        :param sentences: sentence(s) in words；每个 str 都是一个 word，List[str] 表示一个句子
        :return word_ids (list[list[list[int]]]): sentence(s) in indices
        """
        return [[[self.char2id.get(char, self.unk_index) for char in ("{" + word + "}")] for word in sentence] for sentence in sentences]

    def to_tensor(self, sentences: List[List[List[int]]], device: Optional[torch.device]) -> torch.Tensor:
        """ Convert list of sentences (chars) into tensor with necessary padding for shorter sentences.

        :param sentences: list[list[list[int]]]；List[int] 为一个 word(使用 vocab 的 char2id 处理了)，List[List[int]] 为一个句子；
        :param device: device on which to load the tensor, i.e. CPU or GPU

        :return tensor_sentences_padded: tensor of (N, max_sentence_len, max_word_len)  torch.long 是因为 embedding 的 indices 输入必须为 long
        """

        sentences_padded = utils.pad_sentences_char(sentences, self.pad_index)  # 填充单词（以 char 为单位）和句子（以 word 为单位）
        tensor_sentences_padded = torch.tensor(sentences_padded, dtype=torch.long, device=device)  # shape: (N, max_sentence_len, max_word_len)
        return tensor_sentences_padded


class WordVocab:
    """
    参考了 cs224n a5 vocab.py
    把 word 转化为 int
    """
    def __init__(self, word2id: Optional[Dict[str, int]] = None):
        """
        建立词典 word2id[word] = index
        :param word2id: 当从文件读取 word2id 时使用
        """
        if word2id is not None:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.add('<pad>')  # Pad Token
            self.add('<unk>')  # Unknown Token

        self.pad_index = self.word2id['<pad>']
        self.unk_index = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __len__(self) -> int:
        """ Compute number of words in VocabEntry.
        :return len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def add(self, word: str) -> int:
        """
        Add word to VocabEntry, if it is previously unseen.
        :param word: word to add to VocabEntry
        :return index: index that the word has been assigned
        """
        if word not in self.word2id:
            index = len(self.word2id)
            self.word2id[word] = index
            self.id2word[index] = word
            return index
        else:
            return self.word2id[word]

    def words2word_indices(self, sentences: List[List[str]]) -> List[List[int]]:
        """ Convert list of sentences of words into list of list of indices. 把每个 word 换成一个 index
        :param sentences: sentence(s) in words，word 中可能有大写字母
        :return word_ids: sentence(s) in indices
        """
        return [[self.word2id.get(word.lower(), self.unk_index) for word in sentence] for sentence in sentences]

    def word_indices2words(self, sentences_indices: List[List[int]]) -> List[List[str]]:
        """ words2word_indices 的反向操作
        :param sentences_indices: 每个 int 都是一个 word; List[int] 就是一句话
        :return sentences (List[List[str]]): 对所有 index 进行替换，还原句子
        """
        return [[self.id2word[word_index] for word_index in sentence_indices] for sentence_indices in sentences_indices]

    def to_tensor(self, sentences: List[List[int]], device: Optional[torch.device]) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for shorter sentences.

        :param sentences: list of sentences (words)
        :param device: device on which to load the tensor, i.e. CPU or GPU

        :return sentences_var: tensor of (N, max_sentence_len) 句子群变成了 Tensor，sentences_var[第几句][该句的第几个单词] = 单词在 vocab 的 index
        """
        sentences_padded = utils.pad_sentences_word(sentences, self.pad_index)  # 填充句子（以 word 为单位）
        tensor_sentences_padded = torch.tensor(sentences_padded, dtype=torch.long, device=device)
        return tensor_sentences_padded

    @staticmethod
    def to_list(x_sentence_lens: torch.Tensor, x: torch.Tensor) -> List[List[int]]:
        """to_tensor 的反向操作
        :param x_sentence_lens: 每句话的实际长度；shape: (N,)
        :param x: shape: (N, max_sentence_len)  x[第几句][该句的第几个单词] = 单词在 vocab 的 index
        :return:
        """

        n = x_sentence_lens.shape[0]
        result = [x[i][:x_sentence_lens[i]].tolist() for i in range(n)]  # 仅截取有效部分
        return result

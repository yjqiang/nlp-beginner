from collections import Counter
from itertools import chain
from typing import List, Optional, Dict, Tuple
import json

import torch

import data_handle
from utils import utils


class Vocab:
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

    def __len__(self) -> int:
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def add(self, word: str) -> int:
        """
        Add word to VocabEntry, if it is previously unseen.
        @param word: word to add to VocabEntry
        @return index: index that the word has been assigned
        """
        if word not in self.word2id:
            index = len(self.word2id)
            self.word2id[word] = index
            return index
        else:
            return self.word2id[word]

    def get(self, word: str) -> int:
        """
        Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word: word
        @return index: index of the word
        """
        return self.word2id.get(word, self.unk_index)

    def words2indices(self, sentences: List[List[str]]) -> List[List[int]]:
        """ Convert list of sentences of words into list of list of indices. 把每个 word 换成 index
        @param sentences: sentence(s) in words
        @return word_ids: sentence(s) in indices
        """
        return [[self.get(word) for word in sentence] for sentence in sentences]

    def to_input_tensor(self, sentences: List[List[str]], max_sentence_length: int, device: Optional[torch.device]) -> Tuple[torch.Tensor, List[int]]:
        """ Convert list of sentences (words) into tensor with necessary padding for shorter sentences.

        @param sentences: list of sentences (words)
        @param max_sentence_length: 多退少补
        @param device: device on which to load the tensor, i.e. CPU or GPU

        @returns sentences_var: tensor of (batch_size, max_sentence_length) 句子群变成了 Tensor，sentences_var[第几句][该句的第几个单词] = 单词在 vocab 的 index
                 seq_lengths: List of actual lengths for each of the sentences in the batch
        """
        word_ids = self.words2indices(sentences)
        list_sentences, seq_lengths = utils.pad_sentences(word_ids, self.pad_index, max_sentence_length)
        sentences_var = torch.tensor(list_sentences, dtype=torch.long, device=device)
        return sentences_var, seq_lengths

    @staticmethod
    def build_vocab(sentences: List[List[str]], size: Optional[int] = None,  count_cutoff: int = 2) -> 'Vocab':
        """
        根据句子建立词典，把 word 转为 int
        :param sentences: 每句话都是一个 string
        :param size: 截取出现次数很少的忽略掉（给定词典的 max 大小）
        :param count_cutoff: if word occurs n < count_cutoff times, drop the word
        :return:
        """
        # It（Counter） is a collection where elements are stored as dictionary keys and their counts are stored as dictionary values.
        # dictionary keys 就是指 word，count 就是在句子群出现次数
        vocab = Vocab()
        words_count = Counter(chain(*sentences))
        words = [word for word, count in words_count.items() if count >= count_cutoff]

        print(f'number of unique word: {len(words_count)}, number of unique valid word (its count >= {count_cutoff}): {len(words)}')
        if size is not None:
            words = sorted(words, key=words_count.get, reverse=True)[:size]

        for word in words:
            vocab.add(word)
        return vocab

    def save_json(self, file_path: str) -> None:
        """ Save Vocab to file as JSON dump. 保存字典
        @param file_path: file path to vocab file
        """
        json.dump(dict(word2id=self.word2id), open(file_path, 'w'), indent=2)

    @staticmethod
    def load_json(file_path: str) -> 'Vocab':
        """ Load vocabulary from JSON dump.
        @param file_path: file path to vocab file
        @returns Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r'))
        word2id = entry['word2id']
        return Vocab(word2id)


if __name__ == '__main__':
    x_train_orig, _ = data_handle.read_tsv_data0('data/train_split.tsv')

    sentences_ = utils.split_sentences(x_train_orig)  # sentences_ vocab_ 都是为了区别名字，防止 shadowing names defined in outer scopes
    vocab_ = Vocab.build_vocab(sentences_)
    vocab_.save_json('vocab.json')

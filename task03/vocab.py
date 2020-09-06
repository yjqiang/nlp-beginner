from typing import List, Optional, Dict, Tuple

import torch

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

    def convert2indices(self, sentences: List[List[str]]) -> List[List[int]]:
        """ Convert list of sentences (words) into list.

        @param sentences: list of sentences (words)

        @returns word_ids: list of sentences. 把单词转为了 int
        """
        word_ids = self.words2indices(sentences)
        return word_ids

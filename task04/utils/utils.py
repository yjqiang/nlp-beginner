from typing import List


def pad_sentences_char(sentences: List[List[List[int]]], char_pad_token: int) -> List[List[List[int]]]:
    """ Pad list of sentences according to the longest sentence in the batch and longest words in all sentences.
    :param sentences: list of sentences（每个单词用一个 List[int] 代替，每个 int 都是一个 char）, result of `words2charindices()` from `vocab.py`
    :param char_pad_token: index of the character-padding token，即 char "<pad>" 的 index

    :return sentences_padded: Output shape: (N, max_sentence_len, max_word_len)
    """

    sentences_padded = []
    max_word_len = max(len(word) for sentence in sentences for word in sentence)  # 单个单词最长的长度
    max_sentence_len = max(len(sentence) for sentence in sentences)  # 最长句子的长度

    for sentence in sentences:
        sentence_padded = []
        # 处理单词
        for word in sentence:
            word_padded = word + [char_pad_token] * (max_word_len-len(word))  # 短单词之后，填充 "pad" 字符(max_word_len)
            sentence_padded.append(word_padded)

        # 处理句子，[char_pad_token]*max_word_len 就是一个“空单词”，使用空单词来填充句子
        sentence_padded = sentence_padded + [[char_pad_token]*max_word_len] * (max_sentence_len - len(sentence_padded))
        sentences_padded.append(sentence_padded)

    return sentences_padded


def pad_sentences_word(sentences: List[List[int]], word_pad_index: int) -> List[List[int]]:
    """ Pad list of sentences according to the longest sentence in the batch. 填充句子，以 word 为单位
    :param sentences: list of sentences, where each sentence is represented as a list of words
    :param word_pad_index: padding token

    :return sentences_padded: list of sentences where sentences shorter than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length. Output shape: (N, max_sentence_length)


    """
    sentences_padded = []
    max_sentence_len = max(len(sentence) for sentence in sentences)  # 最长句子的长度

    for sentence in sentences:
        # 处理句子，[word_pad_index] 就是一个“空单词”，使用空单词来填充句子
        sentence_padded = sentence + [word_pad_index] * (max_sentence_len - len(sentence))
        sentences_padded.append(sentence_padded)

    return sentences_padded

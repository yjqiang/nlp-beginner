from typing import Iterator, List

import nltk


def split_sentences(sentences: Iterator[str]) -> List[List[str]]:
    """
    把每个原句子进行"小写"操作之后，拆分为 List[str]，元素是 word；然后把多个不同句子即多个 List[str] 组装在一起，构成了 List[List[str]]
    :param sentences: 可以 iterate 的句子群
    :return: 每个句子都是一个 list，然后好多个句子又组成了一个大的 list
    """
    result = []
    for str_sentence in sentences:
        list_sentence = nltk.word_tokenize(str_sentence.lower())  # List[str] 把每个句子写成一个 list，而元素是 word
        result.append(list_sentence)
    return result


def pad_sentences(sentences: List[List[int]], pad_index: int, max_sentence_length: int) -> List[List[int]]:
    """ Pad list of sentences according to the longest sentence in the batch. 填充句子，以 word 为单位
    @param sentences: list of sentences, where each sentence is represented as a list of words
    @param pad_index: padding token
    @param max_sentence_length: 多退少补
    @returns sentences_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sentences_padded = []

    for sentence in sentences:
        len_sentence = len(sentence)
        # 长的截断
        if len_sentence > max_sentence_length:
            sentence_padded = sentence[:max_sentence_length]
        else:
            sentence_padded = [pad_index] * max_sentence_length
            sentence_padded[:len_sentence] = sentence
        sentences_padded.append(sentence_padded)

    return sentences_padded

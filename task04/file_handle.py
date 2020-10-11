"""
直接读写文件
"""

from typing import List, Tuple, Dict, Any
import pickle


def read_data(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    :param path:  eng_bioes.xxx 文件路径

    :return sentences: list of sentences (words)；每个 str 都是一个 word，List[str] 表示一个句子
    :return sentences_labels: 对应每个 word 的 label
    """

    sentences = []  # 每个 str 都是一个 word，List[str] 表示一个句子，List[List[str]] 表示一堆句子
    sentences_labels = []

    with open(path, 'r', encoding='UTF-8') as f:
        sentence_labels = []
        sentence = []  # List[str] 表示一个句子

        for line in f:
            line = line.strip()  # remove the newline character
            if not line:  # 空白行（表示一个句子的结束与下一个句子的开始）
                if sentence:  # 防止空白行连续多个，导致出现空白的句子
                    sentences.append(sentence)
                    sentences_labels.append(sentence_labels)

                    sentence = []
                    sentence_labels = []
            else:
                split_result = line.split()
                assert len(split_result) == 2
                sentence.append(split_result[0])
                sentence_labels.append(split_result[1])

        if sentence:  # 防止最后一行没有空白行，最后一句话录入不到
            sentences.append(sentence)
            sentences_labels.append(sentence_labels)
    return sentences, sentences_labels


# 读取 glove 文件
def read_glove(path: str) -> Tuple[List[List[float]], Dict[str, int]]:
    embedding = []
    word2id = {}
    with open(path, 'r', encoding='UTF-8') as f:
        for index, line in enumerate(f):
            line = line.rstrip()  # remove the newline character
            if line:  # 移除空白行
                list_line = line.split()
                embedding.append([float(value) for value in list_line[1:]])
                word2id[list_line[0]] = index  # word = list_line[0]
    return embedding, word2id


def save_pickle(path: str, data: Any) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)

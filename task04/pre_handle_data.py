"""
把数据进行提取转化，转为 int
"""

from typing import List, Tuple

import main
import file_handle


def read_and_transform_data(path: str) -> Tuple[List[List[List[int]]], List[List[int]], List[List[int]]]:
    """
    读取 eng_bioes.xxx 文件并且转化文件
    :param path:

    :return list_list_list_int_sentences_chars: List[int] 为一个 word，List[List[int]] 为一个句子；
    :return list_list_int_sentences_words: int 为一个 word，List[int] 为一个句子；
    :return list_list_int_sentences_labels: int 为一个 word 的 label，List[int] 为一个句子的所有 label；
    """
    list_list_str_sentences, list_list_str_sentences_labels = file_handle.read_data(path)
    list_list_list_int_sentences_chars = main.CHAR_VOCAB.words2char_indices(list_list_str_sentences)  # 字符级别进行代换 index（未填充）
    list_list_int_sentences_words = main.WORD_VOCAB.words2word_indices(list_list_str_sentences)  # word 级别进行代换 index（未填充）
    list_list_int_sentences_labels = main.LABEL.labels2label_indices(list_list_str_sentences_labels)  # word 级别进行代换 index，即每个单词对应一个 label（未填充）
    print(f'数据集大小： {len(list_list_list_int_sentences_chars)=}, {len(list_list_int_sentences_words)=}, {len(list_list_int_sentences_labels)=}')
    print(f'取样： {list_list_str_sentences[:3]=}, {list_list_str_sentences_labels[:3]=}')
    # 处理数据
    return list_list_list_int_sentences_chars, list_list_int_sentences_words, list_list_int_sentences_labels


if __name__ == '__main__':
    data_path = 'data'
    result = read_and_transform_data(f'{data_path}/eng_bioes.testa')  # Dev；根据论文 Table 2 一致选定
    file_handle.save_pickle(f'{data_path}/eng_bioes_testa.pkl', result)

    result = read_and_transform_data(f'{data_path}/eng_bioes.testb')  # Test；根据论文 Table 2 一致选定
    file_handle.save_pickle(f'{data_path}/eng_bioes_testb.pkl', result)

    result = read_and_transform_data(f'{data_path}/eng_bioes.train')
    file_handle.save_pickle(f'{data_path}/eng_bioes_train.pkl', result)  # Train；根据论文 Table 2 一致选定

"""
把数据进行提取转化，否则炸。。。
"""

from typing import List, Tuple

import main
import file_handle
from utils import utils


def read_and_transform_data(path: str) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    """
    读取 snli 文件并且转化文件
    :param path:
    :return:
    """
    x1_orig, x2_orig, y_orig = file_handle.read_data(path)

    # 处理数据
    x1_sentences_str = utils.split_sentences(x1_orig)
    x1_sentences_int = main.VOCAB.convert2indices(x1_sentences_str)
    x2_sentences_str = utils.split_sentences(x2_orig)
    x2_sentences_int = main.VOCAB.convert2indices(x2_sentences_str)
    return x1_sentences_int, x2_sentences_int, y_orig


if __name__ == '__main__':
    snli_path = 'data/snli_1.0'
    result = read_and_transform_data(f'{snli_path}/snli_1.0_dev.jsonl')
    file_handle.save_pickle(f'{snli_path}/snli_1.0_dev.pkl', result)
    result = read_and_transform_data(f'{snli_path}/snli_1.0_test.jsonl')
    file_handle.save_pickle(f'{snli_path}/snli_1.0_test.pkl', result)
    result = read_and_transform_data(f'{snli_path}/snli_1.0_train.jsonl')
    file_handle.save_pickle(f'{snli_path}/snli_1.0_train.pkl', result)

from typing import Tuple, List, Dict

import pandas as pd


# 读取 train.csv/train_split.tsv/val_split.tsv 文件，并且返回原始数据
def read_tsv_data0(path: str) -> Tuple[pd.Series, pd.Series]:
    dataframe = pd.read_csv(path, delimiter='\t')
    # 特么第一次见 sentence 还有 ' ' 的，还特么可以标注？
    # 当我们在清洗数据时往往会将带有空值的行删除，不论是DataFrame还是Series的index都将不再是连续的索引了，那么这个时候我们可以使用reset_index()方法来重置它们的索引，以便后续的操作。
    # drop=False，此时会获得新的 index 列，而原来的那个连续索引（删除之后，现在非连续了）变成了 index 的数据列，保留了下来。
    dataframe = dataframe[dataframe['Phrase'].str.strip() != ''].reset_index(drop=True)

    sentences = dataframe['Phrase']  # Dtype 为 object，更确切的说，是 str
    labels = dataframe['Sentiment']  # Dtype 为 int64
    return sentences, labels


def save_tsv_data(list_series: List[pd.Series], path: str, sep: str = '\t', index: bool = False) -> None:
    """
    :param list_series: pd.Series 的 list
    :param path: 保存路径
    :param sep: 分隔符
    :param index: write a pandas dataframe to csv file with/without row index 是否会额外有一列专门写 index
    :return:
    """
    dataframe = pd.concat(list_series, axis=1)
    print(dataframe, type(dataframe))
    dataframe.to_csv(path, sep=sep, index=index)


# 读取 glove 文件
def read_glove(path: str) -> Tuple[List[List[float]], Dict[str, int]]:
    embedding = []
    word2id = {}
    with open(path, 'r', encoding='UTF-8') as f:
        for index, line in enumerate(f):
            line = line.rstrip('\n')  # remove the newline character
            if line:  # 移除空白行
                list_line = line.split()
                embedding.append([float(value) for value in list_line[1:]])
                word2id[list_line[0]] = index  # word = list_line[0]
    return embedding, word2id

import json
from typing import List, Tuple, Dict, Any
import pickle


def read_data(path: str) -> Tuple[List[str], List[str], List[int]]:
    """

    :param path:  snli.jsonl 文件路径
    :return:
    """
    label2index = {'contradiction': 0, 'entailment': 1, 'neutral': 2}  # 注意这是 dev test 和 train 共享的，否则不一致就错了

    x1 = []
    x2 = []
    labels = []
    with open(path, 'r') as f:
        for line in f:
            line = json.loads(line)
            if line['gold_label'] == '-':
                # In the case of this unknown label, we will skip the whole datapoint
                continue
            x1.append(line['sentence1'])
            x2.append(line['sentence2'])
            labels.append(label2index[line['gold_label']])
    return x1, x2, labels


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


def save_pickle(path: str, data: Any) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)

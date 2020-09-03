import json
from typing import List, Tuple


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

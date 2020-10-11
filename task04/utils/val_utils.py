"""
负责验证相关的工作
"""
from typing import List, Optional, Tuple


def get_f1_score(tp: int, pred_t: int, real_t: int) -> Tuple[float, str]:
    """
    t=true（真）；f=false（假）
    p=positive（正例）；negative（反例）
    :param tp: 真正例，即预测为正且真值为正
               这里 val 时候是指全部预测正确的实体（“O” 不是实体）
    :param pred_t: pred_t = tp + fp  （tp：真正例，即预测为正且真值为正；fp：假正例，即预测为正但真值为反）
                   这里 val 时候是指全部预测的实体（无论对错）
    :param real_t: real_t = tp + fn  （tp：真正例，即预测为正且真值为正；fn: 假反例，即预测为反但真值为正）
                   这里 val 时候是指全部人工标注的实体
    :return:
    """
    # 查准率 precision = tp/(tp+fp) = tp/pred_t 我预测为正例的结果里面，有多少是真的正例的
    precision = tp / pred_t if pred_t else 0
    # 查全率 recall = tp/(tp+fn) = tp/real_t 所有为真正正例的里面，有多少被我发现了
    recall = tp / real_t if real_t else 0
    # f1 = 1/2 * (1/precision + 1/recall) = 2 * precision * recall / (precision + recall)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return f1, f'f1_score: {f1:.5%}, precision: {precision:.5%}, recall: {recall:.5%}'


def bioes_tag2spans(tags: List[str], ignore_labels: Optional[List[str]] = None) -> List[Tuple[str, Tuple[int, int]]]:
    """
    感谢 fastnlp
    给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'E-singer', 'O', 'O']。
    返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str]
    :param ignore_labels: 在该 list 中的 label 将被忽略
    :return: List[(label, (start, end)), ...]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for index, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]  # 不用 split 是为了防止 “O” 的情况导致错误
        if bmes_tag in ('b', 's'):  # 开始
            spans.append((label, [index, index]))
        elif bmes_tag in ('i', 'e') and prev_bmes_tag in ('b', 'i') and label == spans[-1][0]:  # 属于同一个实体
            spans[-1][1][1] = index
        elif bmes_tag == 'o':
            pass
        else:
            spans.append((label, [index, index]))
        prev_bmes_tag = bmes_tag
    return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels]

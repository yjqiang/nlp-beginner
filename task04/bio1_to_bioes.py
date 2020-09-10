"""
原文件是 BIO1 格式的，转化成 BIOES
"""

from typing import List, Literal


class Element:
    __slots__ = ('word', 'ner_tag_bio1', 'ner_tag_bio2', 'ner_tag_bioes', 'ner_tag_others')

    def __init__(self):
        self.word = None  # 空行的时候，word 为 None
        self.ner_tag_bio1 = 'B/I/O'
        self.ner_tag_bio2 = 'B/I/O'
        self.ner_tag_bioes = 'B/I/O/E/S'
        self.ner_tag_others = 'PER/ORG/LOC/MISC'

    @staticmethod
    def initial_from_conll_2003(line: List[str]) -> 'Element':
        """
        这里我们不需要其他数据，所以仅保留 word 和 命名体标注（3 个标注中的最后一个标注）。
        :param line: conll_2003 原文中，每行或者为空字符串，或者为 'word xx B-xx/I-xx/O B-xx/I-xx/O' 格式 split 开来（eg: ['CRICKET', 'NNP', 'I-NP', 'O']）；第一行是 '-DOCSTART- xxx xx O'
        :return:
        """
        element = Element()

        # 空行不处理
        if not line:
            element.word = None  # 空行的时候，word 为 None
            return element

        element.word, _, _, org_ner_tags = line  # tags 由两个部分组成，前面是 B/I/O，后面是 PER/ORG/LOC/MISC，用 '-' 连接
        element.ner_tag_bio1 = org_ner_tags[0]
        element.ner_tag_others = org_ner_tags[2:]  # 扣掉 '-'；以及当 tag 为 O 时候，ner_tag_others 为 ''
        return element

    def as_str(self, tag_type: Literal['ner_tag_bio1', 'ner_tag_bio2', 'ner_tag_bioes'], end: str = '\n') -> str:
        if self.word is not None:
            the_tag = getattr(self, tag_type)
            return f'{self.word} {the_tag}-{self.ner_tag_others}{end}' if the_tag != 'O' else f'{self.word} O{end}'
        return end  # 空白字符串

    @property
    def is_empty(self) -> bool:
        return self.word is None


def bio2_to_bioes(elements: List[Element]) -> None:
    """
    :param elements:
    :return:
    """
    for i, element in enumerate(elements):
        if element.ner_tag_bio2 == 'O':
            element.ner_tag_bioes = 'O'
        elif element.ner_tag_bio2 == 'B':
            # 非 single word 实体 （BIO2 实体开始必定是 B）
            if i + 1 < len(elements) and elements[i + 1].ner_tag_bio2 == 'I':
                element.ner_tag_bioes = 'B'
            # 包含了本条目为最后一条的情况
            else:
                element.ner_tag_bioes = 'S'
        else:  # elif element.ner_tag_bio2 == 'I':
            # 此实体至少包含前一个 word、本 word 和 下一个 word
            if i + 1 < len(elements) and elements[i + 1].ner_tag_bio2 == 'I':
                element.ner_tag_bioes = 'I'
            # 本 word 就是这个实体的结尾了
            else:
                element.ner_tag_bioes = 'E'


def bio1_to_bio2(elements: List[Element]) -> None:
    """
    :param elements:
    :return:
    """
    assert elements[0].ner_tag_bio1 == 'O'

    for i, element in enumerate(elements):
        if element.ner_tag_bio1 == 'O':
            element.ner_tag_bio2 = 'O'
        elif element.ner_tag_bio1 == 'B':
            element.ner_tag_bio2 = 'B'

        # 剩下的都是 I 的情况了(element.ner_tag_bio1)！！！

        # 本实体至少包含了前一个 word 和本个 word
        # eg： 小华南小明（三人出去玩） -> BIO1: I-PER I-PER B-PER B-PER I-PER (....)    BIO2: B-PER I-PER B-PER B-PER I-PER (....)
        # eg: Bill works for Bank of America. -> BIO1: I-PER O O I-ORG I-ORG I-ORG.   BIO2: B-PER O O B-ORG I-ORG I-ORG ....
        elif i > 0 and elements[i - 1].ner_tag_others == elements[i].ner_tag_others:
            element.ner_tag_bio2 = 'I'
        # 出现在首次的 I，即本 word 是新的实体的开始 (elements[i - 1].ner_tag_others != elements[i].ner_tag_others)
        # elements[i - 1].ner_tag_bio1 == 'O' 时候，elements[i - 1].ner_tag_others == '' 也包含在这里来
        else:
            element.ner_tag_bio2 = 'B'


def conll_2003_to_bioes(in_path: str, out_path: str):
    """
    转换格式；原 List 每行或者为空字符串，或者为 'word xx B-xx/I-xx/O B-xx/I-xx/O' 格式
    :param in_path: 输入
    :param out_path: 输出
    :return:
    """
    with open(in_path, 'r', encoding="utf8") as f:
        lines = f.read().splitlines()  # 读取文件，返回 List，每个元素都是一行文本，且没有换行符号；空行为 ''

    elements = [Element.initial_from_conll_2003(line.split()) for line in lines]  # '' -> []；'CRICKET NNP I-NP O' -> ['CRICKET', 'NNP', 'I-NP', 'O']
    bio1_to_bio2(elements)
    bio2_to_bioes(elements)
    lines = [element.as_str('ner_tag_bioes') for element in elements]

    with open(out_path, 'w', encoding="utf8") as f:
        f.writelines(lines)


if __name__ == '__main__':
    conll_2003_to_bioes('data/eng.testa', 'data/eng_bioes.testa')
    conll_2003_to_bioes('data/eng.testb', 'data/eng_bioes.testb')
    conll_2003_to_bioes('data/eng.train', 'data/eng_bioes.train')

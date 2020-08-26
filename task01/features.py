import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class BagOfWordFeature:
    def __init__(self, n_gram: int = 1):
        self._count_vectorizer = CountVectorizer(ngram_range=(n_gram, n_gram))

    def fit(self, x_train_orig: pd.Series) -> np.ndarray:
        """
        习得词典，并返回 x_train_orig 的 词袋特征
        :param x_train_orig: Dtype 为 object，更确切的说，是 str；每个元素都是一个句子 sentence (str)，即 x_train_orig = sentences
        :return: shape 为 (N, feature_size=词典大小)
        """
        x_train_ = self._count_vectorizer.fit_transform(x_train_orig)  # 获得 词典，同时返回 输入 的处理结果(type = scipy.sparse.csr_matrix)
        return x_train_.toarray()  # 使用 toarray 转为矩阵形式，行列坐标表示(第几句，词典中第几个单词)，对应元素为 在本句子中出现了几次

    def transform(self, x_orig: pd.Series) -> np.ndarray:
        """
        不改变词典！！返回 x_orig 的 词袋特征；基本与 fit 函数类似
        :param x_orig:  Dtype 为 object，更确切的说，是 str；每个元素都是一个句子 sentence (str)，即 x = sentences
        :return: shape 为 (N, feature_size=词典大小)
        """
        x_ = self._count_vectorizer.transform(x_orig)
        return x_.toarray()

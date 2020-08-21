"""
单层的实现，方便拼接
"""
from typing import Tuple

import numpy as np


class LinearLayer:
    """
    x @ w + b
    x: (N, input_features)
    w: (input_features, out_features)
    b: (out_features,)
    """
    def __init__(self, input_features: int, out_features: int, weight_scale: float = 1e-3):
        self.w = np.random.randn(input_features, out_features) * weight_scale  # 随机初始化
        self.b = np.zeros(out_features)
        self.x = None  # 用于反向传播

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        :param x: shape: (N, input_features)
        :return: shape: (N, out_features)
        """
        self.x = x
        return LinearLayer.for_grad_check(self.w, self.b, x)

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        反向传播
        :param dout: shape (N, out_features)
        :return:
        """
        dx = dout @ self.w.T  # shape: (N, input_features)
        dw = self.x.T @ dout  # shape: (input_features, out_features)
        db = np.sum(dout, axis=0)
        return dx, dw, db

    @staticmethod
    def for_grad_check(w: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        前向传播的相同表示，用于梯度检查
        :param w: shape: (input_features, out_features)
        :param b: shape: (out_features,)
        :param x: shape: (N, input_features)
        :return:
        """
        return x @ w + b


class CrossEntropyLossLayer:
    """
    交叉熵
    """
    def __init__(self):
        self.dx = None  # 用于反向传播；最后一步了，直接就能求了

    def forward(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        前向传播
        :param x: shape: (N, num_of_classes) x 就是 scores
        :param y: shape: (N, )   y[i] 是 x[i] 的分类真值，且 0 <= y[i] < num_of_classes
        :return: loss
        """
        N = x.shape[0]

        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_x)
        ps = np.sum(exp_scores, axis=1).reshape((-1, 1))
        loss = (- np.sum(shifted_x[np.arange(N), y]) + np.sum(np.log(ps))) / N

        # 顺手求 dx 因为，exp_scores 和 ps 直接用很舒服；而且最后一步了，直接就能求了
        dx = exp_scores / ps
        dx[np.arange(N), y] -= 1
        dx /= N
        self.dx = dx
        return loss

    def backward(self) -> np.ndarray:
        """
        反向传播
        :return:
        """
        return self.dx

    @staticmethod
    def for_grad_check(x: np.ndarray, y: np.ndarray) -> float:
        """
        前向传播的相同表示，用于梯度检查（我这里也想复用，可 forward 中 exp_scores 和 ps 这些都得用到阿）
        :param x: shape: (N, num_of_classes) x 就是 scores
        :param y: shape: (N, )   y[i] 是 x[i] 的分类真值，且 0 <= y[i] < num_of_classes
        :return: loss
        """
        N = x.shape[0]

        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_x)
        ps = np.sum(exp_scores, axis=1).reshape((-1, 1))
        loss = (- np.sum(shifted_x[np.arange(N), y]) + np.sum(np.log(ps))) / N

        return loss

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
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        :param x: shape: (N, input_features)
        :return: shape: (N, out_features)
        """
        self.x = x
        return x @ self.w + self.b

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


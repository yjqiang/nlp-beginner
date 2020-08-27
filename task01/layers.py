"""
单层的实现，方便拼接
"""
from cupy_or_numpy import np


class Variable:
    """
    模仿曾经的 pytorch 的 Variable
    """
    def __init__(self, value: np.ndarray):
        self.value = value
        self.dvalue = None


class LinearLayer:
    """
    x @ w + b
    x: (N, input_features)
    w: (input_features, out_features)
    b: (out_features,)
    """
    def __init__(self, input_features: int, out_features: int, weight_scale: float = 1e-3):
        self.w = Variable(np.random.randn(input_features, out_features) * weight_scale)  # 随机初始化
        self.b = Variable(np.zeros(out_features))
        self.parameters = [self.w, self.b]
        self.x = None  # 用于反向传播

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        :param x: shape: (N, input_features)
        :return: shape: (N, out_features)
        """
        self.x = x
        return x @ self.w.value + self.b.value

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播
        :param dout: shape (N, out_features)
        :return:
        """
        dx = dout @ self.w.value.T  # shape: (N, input_features)
        dw = self.x.T @ dout  # shape: (input_features, out_features)
        db = np.sum(dout, axis=0)
        self.w.dvalue = dw
        self.b.dvalue = db
        return dx

    def grad_check(self, parameter_name: str, value: np.ndarray, **kwargs) -> np.ndarray:
        """
        辅助梯度检查，负责在更改任一变量后，输出前向传播结果
        :param parameter_name: 例如 'w' 'b'， x 通过 kwargs 进行输入；为 '' 时候不会改变参数
        :param value:
        :param kwargs: 负责 forward() 函数的输入
        :return:
        """
        if parameter_name:
            getattr(self, parameter_name).value = value
        return self.forward(**kwargs)


class CrossEntropyLossLayer:
    """
    交叉熵
    """
    def __init__(self):
        self.dx = None  # 用于反向传播；最后一步了，直接就能求了
        self.parameters = []

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

    def grad_check(self, parameter_name: str, value: np.ndarray, **kwargs) -> float:
        """
        辅助梯度检查，负责在更改任一变量后，输出前向传播结果
        :param parameter_name: 始终为 '' 因为 CrossEntropyLossLayer 没有参数
        :param value:
        :param kwargs: 负责 forward() 函数的输入
        :return:
        """
        if parameter_name:
            getattr(self, parameter_name).value = value
        return self.forward(**kwargs)

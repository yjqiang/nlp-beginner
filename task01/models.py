import math

import numpy as np
from tqdm import tqdm

import layers
from utils import general_utils
from cupy_or_numpy import np as np_


class Model:
    def __init__(self, input_features: int, out_features: int):
        """
        :param input_features: 输入层的特征数（维度）
        :param out_features: The number of classes C.
        """
        self.linear_layer = layers.LinearLayer(input_features=input_features, out_features=out_features)
        self.cross_entropy_loss_layer = layers.CrossEntropyLossLayer()
        self.parameters = self.linear_layer.parameters + self.cross_entropy_loss_layer.parameters

    def get_scores(self, x: np_.ndarray) -> np.ndarray:
        """
        求分数，是 softmax 的前一层
        :param x: shape: (N, input_features)
        :return scores: shape: (N, out_features)
        """
        output_linear_layer = self.linear_layer.forward(x)  # 这就是 scores，shape: (N, out_features)
        return output_linear_layer

    def forward(self, x: np_.ndarray, y: np_.ndarray, reg: float = 0.0) -> float:
        """
        前向传播
        :param x: shape: (N, input_features)
        :param y: shape: (N, )   y[i] 是 x[i] 的分类真值，且 0 <= y[i] < num_of_classes
        :param reg: l2 的权重
        :return:
        """

        scores = self.get_scores(x)  # shape: (N, out_features)
        loss = self.cross_entropy_loss_layer.forward(scores, y)  # float
        return loss

    def backward(self) -> np.ndarray:
        """
        反向传播
        :return: 这个返回值没卵用的
        """
        dout = self.cross_entropy_loss_layer.backward()
        return self.linear_layer.backward(dout)

    def grad_check(self, variable_name: str, parameter_name: str, value: np.ndarray, **kwargs) -> float:
        """
        辅助梯度检查，负责在更改任一变量后，输出前向传播结果
        :param variable_name: 例如 'linear_layer' 'cross_entropy_loss_layer'；为 '' 时候不会改变参数
        :param parameter_name: 例如 'w' 'b'， x 通过 kwargs 进行输入
        :param value:
        :param kwargs: 负责 forward() 函数的输入
        :return:
        """
        if variable_name:
            getattr(getattr(self, variable_name), parameter_name).value = value
        return self.forward(**kwargs)

    def check_accuracy(self, x: np_.ndarray, y: np_.ndarray) -> None:
        """
        检查准确率
        :param x: shape: (N, input_features)
        :param y: shape: (N, )   y[i] 是 x[i] 的分类真值，且 0 <= y[i] < num_of_classes
        :return:
        """
        scores = self.get_scores(x)  # shape: (N, out_features=类别数目)
        pred_y = np_.argmax(scores, axis=1)  # shape: (N2, )
        accuracy = float(np_.sum(y == pred_y) / y.shape[0])  # cupy.sum https://docs.cupy.dev/en/stable/reference/difference.html#reduction-methods 显式转化
        print(f"当前准确率为: {accuracy:.5%}", )

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, batch_size: int, epoch_num: int, learning_rate: float, reg: float = 0.0) -> None:
        """
        训练
        :param x_train: 训练集输入特征 forward；shape: (N1, input_features)
        :param y_train: 训练集真值；shape: (N1, )
        :param x_val: 验证集输入特征；shape: (N2, input_features)
        :param y_val: 验证集真值；shape: (N2, )
        :param batch_size:
        :param epoch_num: 训练集中全部的数据被用过一次，叫一个epoch
        :param learning_rate: 学习率
        :param reg:
        :return:
        """
        for epoch in range(epoch_num):
            self.train_for_epoch(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, batch_size=batch_size, learning_rate=learning_rate, reg=reg)

    def train_for_epoch(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, batch_size: int, learning_rate: float, reg: float = 0.0) -> None:
        """
        epoch:训练集中全部的数据被用过一次，叫一个epoch

        :param x_train: 训练集输入特征 forward；shape: (N1, input_features)
        :param y_train: 训练集真值；shape: (N1, )
        :param x_val: 验证集输入特征；shape: (N2, input_features)
        :param y_val: 验证集真值；shape: (N2, )
        :param batch_size:
        :param learning_rate: 学习率
        :param reg:
        :return:
        """
        x_val = np_.asarray(x_val)
        y_val = np_.asarray(y_val)

        n_minibatches = math.ceil(x_train.shape[0] / batch_size)
        loss_meter = general_utils.AverageMeter()

        with tqdm(total=n_minibatches) as prog:
            for iterator, (cur_x_train, cur_y_train) in enumerate(general_utils.get_minibatches([x_train, y_train], batch_size)):  # 每次一个 batch
                cur_x_train = np_.asarray(cur_x_train)
                cur_y_train = np_.asarray(cur_y_train)
                loss = self.forward(cur_x_train, cur_y_train, reg)
                self.backward()

                for param in self.parameters:
                    param.value -= learning_rate * param.dvalue

                if iterator % 100 == 0 and iterator:  # iterator % 100 == 0 且 iterator != 0
                    print(f'当前训练集的 loss: {loss}')
                    self.check_accuracy(x_val, y_val)

                prog.update(1)  # tqdm 更新进度条
                loss_meter.update(loss)

        self.check_accuracy(x_val, y_val)
        print(f"本 EPOCH 训练的平均 LOSS: {loss_meter.avg}")

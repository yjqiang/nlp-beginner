"""
assignment1/cs231n/gradient_check.py
"""
from typing import Callable
from random import randrange

import numpy as np


def check_rel_error(x: np.ndarray, y: np.ndarray) -> float:
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))  # 1e-8 防止除零操作


def eval_numerical_gradient_array(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, analytic_grad: np.ndarray, df: np.ndarray, h: float = 1e-5) -> None:
    """
    Evaluate a numeric gradient for a function that accepts a numpy array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    for x_index in np.ndindex(x.shape):  # 列出 x 的所有坐标
        cur_val = x[x_index]
        x[x_index] = cur_val + h
        pos = f(x).copy()  # 计算 f(x + h)
        x[x_index] = cur_val - h
        neg = f(x).copy()  # 计算 f(x - h)
        x[x_index] = cur_val  # reset

        # 偏导展开 △loss = Σ(dout_ij * △out_ij) + others(d*△) = Σ(dx_i'j' * △x_i'j') + others'(d*△)
        # others'(d*△) 包含了 others(d*△) 的全部 以及 Σ(dout_ij * △out_ij) 除 Σ(dx_i'j' * △x_i'j') 之外的拆分(eg: wx+b=out, x 就是 x 时，除了 dx 还有 dw db)
        # 我们固定所有的数据 除了某个 x_i'j'，更改其大小，两边的 others(d*△) “抵消”之后，左侧仅有 Σ(dout_ij * △out_ij) ，右侧非 0 的 △ 仅有 △x_i'j'
        # 所有求得 Σ(dout_ij * △out_ij) = △x_i'j' * dx_i'j'
        # out 也就是 f
        grad[x_index] = np.sum((pos - neg) * df) / (2 * h)

    print(f'relative error: {check_rel_error(grad, analytic_grad):e}')


def grad_check_sparse(f: Callable[[np.ndarray], float], x: np.ndarray, analytic_grad: np.ndarray, num_checks: int = 10, h: float = 1e-5) -> None:
    """
    sample a few random elements and only return numerical in this dimensions. 即随机去挑选某几个元素，进行偏导的验证
    """

    for _ in range(num_checks):
        x_index = tuple([randrange(m) for m in x.shape])  # 随机选择 x 的某一个坐标
        # 求偏导: ∂result / ∂x[x_index]
        cur_val = x[x_index]
        x[x_index] = cur_val + h  # increment by h
        f_x_plus_h = f(x)  # 计算 f(x + h)
        x[x_index] = cur_val - h  # increment by h
        f_x_minus_h = f(x)  # 计算 f(x - h)
        x[x_index] = cur_val  # reset
        dx_numerical = (f_x_plus_h - f_x_minus_h) / (2 * h)  # 估算的偏导

        dx_analytic = analytic_grad[x_index]  # 要进行梯度检验的偏导
        rel_error = abs(dx_numerical - dx_analytic) / max(1e-8, (abs(dx_numerical) + abs(dx_analytic)))  # 1e-8 防止除零操作
        print(f'numerical: {dx_numerical:f} analytic: {dx_analytic:f}, relative error: {rel_error:e}')

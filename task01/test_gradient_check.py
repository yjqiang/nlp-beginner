import numpy as np

import layers
import gradient_check
import models


input_features = 87
out_features = 57
N = 10

x = np.random.randn(N, input_features)
w = np.random.randn(input_features, out_features)
b = np.random.randn(out_features)

# 覆盖 layer 初始化时候自己的值
layer = layers.LinearLayer(input_features, out_features)
layer.w.value = w
layer.b.value = b
out = layer.forward(x)

dout = np.random.randn(*out.shape)
dx = layer.backward(dout)
dw = layer.w.dvalue
db = layer.b.dvalue

gradient_check.eval_numerical_gradient_array(lambda w_: layer.grad_check('w', w_, x=x), w, dw, dout)
gradient_check.eval_numerical_gradient_array(lambda b_: layer.grad_check('b', b_, x=x), b, db, dout)
gradient_check.eval_numerical_gradient_array(lambda x_: layer.grad_check('', x_, x=x_), x, dx, dout)


########################################################################

scores = np.random.randn(N, 21)
y = np.random.randint(21, size=(N, ))  # 正确分类

layer = layers.CrossEntropyLossLayer()
out = layer.forward(scores, y)

dx = layer.backward()

gradient_check.grad_check_sparse(lambda x_: layer.grad_check('', x_, x=x_, y=y), scores, dx)

########################################################################
# 测试整个模型的梯度
model = models.Model(input_features, out_features)
loss = model.forward(x, y)

dx = model.backward()
dw = model.linear_layer.w.dvalue
db = model.linear_layer.b.dvalue

gradient_check.grad_check_sparse(lambda w_: model.grad_check('linear_layer', 'w', w_, x=x, y=y), model.linear_layer.w.value, dw)
gradient_check.grad_check_sparse(lambda b_: model.grad_check('linear_layer', 'b', b_, x=x, y=y), model.linear_layer.b.value, db)
gradient_check.grad_check_sparse(lambda x_: model.grad_check('', '', x_, x=x_, y=y), x, dx)

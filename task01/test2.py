import torch

x1 = torch.randn(1, requires_grad=True)
a = x1 + 2
b = x1 * a

# b.grad_fn.next_functions 包含 a(AccumulateGrad) 和 c(AddBackward0)
print(f'{b.grad_fn.next_functions=}')
assert b.grad_fn.next_functions[0][0].variable is x1 and b.grad_fn.next_functions[1][0] is a.grad_fn
print()

# c.grad_fn.next_functions 包括 a(AccumulateGrad) 和 2(None)
print(f'{a.grad_fn.next_functions=}')
assert a.grad_fn.next_functions[0][0].variable is x1 and a.grad_fn.next_functions[1][0] is None
print('-' * 150)
###############################################################################################################################################
x1 = torch.randn(1, requires_grad=False)
a = x1 * (x1 + 2)
x2 = torch.randn(1, requires_grad=True)
x3 = torch.randn(1, requires_grad=True)
b = a * x2
c = b * x3

assert not a.requires_grad and a.is_leaf() and a.grad_fn is None
assert b.requires_grad and not b.is_leaf() and b.grad_fn is not None
assert x2.requires_grad and x2.is_leaf() and x2.grad_fn is None

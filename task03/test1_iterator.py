import torch
import my_iterator


x1 = [
    torch.tensor([1, ], dtype=torch.float),
    torch.tensor([1, 2], dtype=torch.float),
    torch.tensor([1, 2, 3], dtype=torch.float),
    torch.tensor([1, 2, 3, 4], dtype=torch.float)
]

x2 = [
    torch.tensor([-1, ], dtype=torch.float),
    torch.tensor([-1, -2], dtype=torch.float),
    torch.tensor([-1, -2, -3], dtype=torch.float),
    torch.tensor([-1, -2, -3, -4], dtype=torch.float)
]

y = torch.tensor([101, 102, 103, 104], dtype=torch.int8)

train_dataloader = my_iterator.MyDataLoader(x1, x2, y, shuffle=False, batch_size=3)
for x in train_dataloader:
    print(x)

for x in train_dataloader:
    print(x)

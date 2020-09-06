import torch
from torch.utils.data import DataLoader

import file_handle
from dataset import MyDataset
from modules import esim

import main


PATH = 'saved_model/epoch_3_accuracy_86.62873%.pt'

model = esim.ESIM(embedding=main.EMBEDDING, embedding_size=main.EMBEDDING_SIZE, hidden_size=main.HIDDEN_SIZE, class_num=main.CLASS_NUM)
model.load_state_dict(torch.load(PATH))
model.to(main.DEVICE)


x1_test_list_list_int, x2_test_list_list_int, y_test_orig = file_handle.load_pickle('data/snli_1.0/snli_1.0_test.pkl')

# 处理数据
# torch.long 是因为我们需要把用这个作为 index，转化为 embeddings
x1_test_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(sentence, dtype=torch.long, device=None) for sentence in x1_test_list_list_int],
                                                 batch_first=True,
                                                 padding_value=main.VOCAB.pad_index)
x1_test_seq_lengths = torch.tensor([len(sentence) for sentence in x1_test_list_list_int])

x2_test_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(sentence, dtype=torch.long, device=None) for sentence in x2_test_list_list_int],
                                                 batch_first=True,
                                                 padding_value=main.VOCAB.pad_index)
x2_test_seq_lengths = torch.tensor([len(sentence) for sentence in x2_test_list_list_int])
y_test = torch.tensor(y_test_orig)
print(x1_test_tensor.shape, x2_test_tensor.shape, torch.max(x1_test_seq_lengths), torch.max(x2_test_seq_lengths))

# 装载到 DataLoader
test_dataset = MyDataset(x1_test_tensor, x1_test_seq_lengths, x2_test_tensor, x2_test_seq_lengths, y_test)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=main.BATCH_SIZE)
print(main.evaluate(model, test_dataloader, test_dataset))

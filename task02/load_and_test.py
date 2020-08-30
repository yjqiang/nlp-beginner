import torch
from torch.utils.data import DataLoader

import main
import data_handle
from dataset import MyDataset
from utils import utils
from modules.cnn import Model


PATH = 'saved_model/epoch_5_accuracy_65.77639%.pt'

model = Model(num_embeddings=len(main.VOCAB), padding_idx=main.VOCAB.pad_index, embedding_size=main.EMBEDDING_SIZE, list_filter_nums=main.LIST_FILTER_NUMS, list_window_sizes=main.LIST_WINDOW_SIZES,
              max_sentence_length=main.MAX_SENTENCE_LENGTH, dropout_p=main.DROPOUT_P, class_num=main.CLASS_NUM, list_paddings=main.LIST_PADDINGS)
model.load_state_dict(torch.load(PATH))
model.to(main.DEVICE)


x_test_orig, y_test_orig = data_handle.read_tsv_data0('data/val_split.tsv')

x_test_sentences = utils.split_sentences(x_test_orig)
x_test = main.VOCAB.to_input_tensor(x_test_sentences, main.MAX_SENTENCE_LENGTH, device=None)  # 一口气全丢到 cuda 里面，你不怕炸么
y_test = torch.tensor(y_test_orig)
x_test, y_test = x_test[x_test.shape[0] // 2:], y_test[y_test.shape[0] // 2:]  # 在 task01 数据基础上，把 val 分劈出来一个 test
print(f'{x_test.shape=}, {y_test.shape=}')

# 装载到 DataLoader
test_dataset = MyDataset(x=x_test, y=y_test)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=main.BATCH_SIZE)
print(main.evaluate(model, test_dataloader, test_dataset))

import torch
from torch.utils.data import DataLoader

import cnn_main
import data_handle
from dataset import MyDataset
from utils import utils
from modules.cnn import Model


PATH = f'{cnn_main.PATH}/epoch_13_accuracy_66.26852%.pt'

model = Model(vocab=cnn_main.VOCAB, embedding_size=cnn_main.EMBEDDING_SIZE, list_filter_nums=cnn_main.LIST_FILTER_NUMS, list_window_sizes=cnn_main.LIST_WINDOW_SIZES,
              max_sentence_length=cnn_main.MAX_SENTENCE_LENGTH, dropout_p=cnn_main.DROPOUT_P, class_num=cnn_main.CLASS_NUM, list_paddings=cnn_main.LIST_PADDINGS, embedding=cnn_main.EMBEDDING)
model.load_state_dict(torch.load(PATH))
model.to(cnn_main.DEVICE)


x_test_orig, y_test_orig = data_handle.read_tsv_data0('data/val_split.tsv')

x_test_sentences = utils.split_sentences(x_test_orig)
x_test, seq_lengths_test = cnn_main.VOCAB.to_input_tensor(x_test_sentences, cnn_main.MAX_SENTENCE_LENGTH, device=None)  # 一口气全丢到 cuda 里面，你不怕炸么
y_test = torch.tensor(y_test_orig)
x_test, y_test, seq_lengths_test = x_test[:x_test.shape[0] // 2], y_test[:y_test.shape[0] // 2], seq_lengths_test[:len(seq_lengths_test) // 2]  # task01 数据基础上，把 val 再劈一个 test，见 cnn_load_and_test.py
print(f'{x_test.shape=}, {y_test.shape=}')

# 装载到 DataLoader
test_dataset = MyDataset(x=x_test, y=y_test, seq_lengths=seq_lengths_test)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=cnn_main.BATCH_SIZE)
print(cnn_main.evaluate(model, test_dataloader, test_dataset))

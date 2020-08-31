import torch
from torch.utils.data import DataLoader

import rnn_main
import data_handle
from dataset import MyDataset
from utils import utils
from modules.rnn import Model


PATH = f'{rnn_main.PATH}/epoch_6_accuracy_66.87343%.pt'

model = Model(vocab=rnn_main.VOCAB, embedding_size=rnn_main.EMBEDDING_SIZE,
              hidden_size=rnn_main.HIDDEN_SIZE, num_layers=rnn_main.NUM_LAYERS, class_num=rnn_main.CLASS_NUM, dropout_p=rnn_main.DROPOUT_P, embedding=rnn_main.EMBEDDING)
model.load_state_dict(torch.load(PATH))
model.to(rnn_main.DEVICE)


x_test_orig, y_test_orig = data_handle.read_tsv_data0('data/val_split.tsv')

x_test_sentences = utils.split_sentences(x_test_orig)
x_test, seq_lengths_test = rnn_main.VOCAB.to_input_tensor(x_test_sentences, rnn_main.MAX_SENTENCE_LENGTH, device=None)  # 一口气全丢到 cuda 里面，你不怕炸么
y_test = torch.tensor(y_test_orig)
x_test, y_test, seq_lengths_test = x_test[:x_test.shape[0] // 2], y_test[:y_test.shape[0] // 2], seq_lengths_test[:len(seq_lengths_test) // 2]  # task01 数据基础上，把 val 再劈一个 test，见 cnn_load_and_test.py
print(f'{x_test.shape=}, {y_test.shape=}')

# 装载到 DataLoader
test_dataset = MyDataset(x=x_test, y=y_test, seq_lengths=seq_lengths_test)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=rnn_main.BATCH_SIZE)
print(rnn_main.evaluate(model, test_dataloader, test_dataset))

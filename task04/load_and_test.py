import torch
from torch.utils.data import DataLoader

import file_handle
from dataset import MyDataset
from modules import ner

import main


PATH = 'saved_model/crf_epoch_41_f1_94.88129%.pt'

model = ner.NER(e_char=main.E_CHAR, e_word=main.E_WORD, cnn_window_size=main.CNN_WINDOW_SIZE, cnn_filter_num=main.CNN_FILTER_NUM, cnn_padding=main.CNN_PADDING,
                dropout_p=0.5, hidden_size=main.LSTM_HIDDEN_SIZE, num_layers=main.LSTM_NUM_LAYERS, vocab=main.CHAR_VOCAB, label=main.LABEL, word_embedding=main.EMBEDDING, use_crf=main.USE_CRF)
model.load_state_dict(torch.load(PATH))
model.to(main.DEVICE)

data_path = 'data'
x_test_list_list_list_int_sentences_chars, x_test_list_list_int_sentences_words, y_test_list_list_str_sentences_labels = file_handle.load_pickle(f'{data_path}/eng_bioes_testb.pkl')

# 处理数据(补上 0)
x_test_sentences_chars = main.CHAR_VOCAB.to_tensor(x_test_list_list_list_int_sentences_chars, device=None)  # shape: (N, max_sentence_len, max_word_len)
x_test_sentences_words = main.WORD_VOCAB.to_tensor(x_test_list_list_int_sentences_words, device=None)  # shape: (N, max_sentence_len)
x_test_sentence_lens = torch.tensor([len(sentence) for sentence in x_test_list_list_list_int_sentences_chars])  # shape: (N,)
y_test_labels = main.LABEL.to_tensor(y_test_list_list_str_sentences_labels, device=None)  # shape: (N, max_sentence_len1)

print(f'{x_test_sentences_chars.shape=}, {x_test_sentences_words.shape=}, {y_test_labels.shape=}, {torch.max(x_test_sentence_lens)=}')

# 装载到 DataLoader
test_dataset = MyDataset(x_sentences_chars=x_test_sentences_chars, x_sentences_words=x_test_sentences_words, x_sentences_lens=x_test_sentence_lens, y=y_test_labels)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=40)  # 验证集的 batch_size 无所谓
_, scores = main.evaluate(model, test_dataloader, verbose=True)
print(scores)

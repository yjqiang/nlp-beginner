from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import file_handle
from vocab import CharVocab, WordVocab
from label import Label
from dataset import MyDataset
from modules import ner
from utils import val_utils


USE_CRF = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 50
BATCH_SIZE = 10  # 见论文 3.2 Optimization Algorithm Parameter optimization is performed with minibatch stochastic gradient descent (SGD) with batch size 10

# 见论文 3.2 Optimization Algorithm
# optimizer: SDG
LEARNING_RATE = 0.015  # η0 = 0.01 for POS tagging, and 0.015 for NER
DECAY_RATE = 0.05  # the learning rate is updated on each epoch of training as ηt = η0/(1 + ρt), with decay rate ρ = 0.05 and t is the number of epoch completed
MOMENTUM = 0.9  # momentum 0.9
CLIP = 5  # To reduce the effects of “gradient exploding”, we use a gradient clipping of 5.0


# 见论文 Table 1
E_CHAR = 30  # 3.1 Parameter Initialization   we set dim = 30
E_WORD = 100  # 3.1 Parameter Initialization We use Stanford’s publicly available GloVe 100-dimensional embeddings1 trained on 6 billion words from Wikipedia and web text (Pennington et al., 2014)

LSTM_HIDDEN_SIZE = 400  # Table 1: state size 200 200
CNN_WINDOW_SIZE = 3  # Table 1: window size 3 3
CNN_PADDING = 2
CNN_FILTER_NUM = 30  # Table 1: number of filters 30 30
LSTM_NUM_LAYERS = 1


print('发现预训练的词向量，加载...')
list_embedding, word2id = file_handle.read_glove("data/glove.6B.100d.txt")
# glove 缺失 pad 和 unk，手动帮助生成
pad_index = len(list_embedding)
list_embedding.append(torch.zeros(E_WORD, dtype=torch.float).tolist())
word2id['<pad>'] = pad_index
unk_index = len(list_embedding)
list_embedding.append(torch.randn(E_WORD).tolist())
word2id['<unk>'] = unk_index

WORD_VOCAB = WordVocab(word2id)
tensor_embedding = torch.tensor(list_embedding).to(DEVICE)
assert tensor_embedding.shape[1] == E_WORD  # 确保维度一致
EMBEDDING = nn.Embedding.from_pretrained(tensor_embedding, freeze=False, padding_idx=WORD_VOCAB.pad_index)  # the tensor does get updated in the learning process!! 微调（其实就是个初始化）
print('预训练的词向量加载完毕')

CHAR_VOCAB = CharVocab()
LABEL = Label()


def evaluate(model: ner.NER, dataloader: DataLoader, verbose: bool = False) -> Tuple[float, str]:
    # tp: 真正例，即预测为正且真值为正
    # 这里 val 时候是指全部预测正确的实体（“O” 不是实体）
    tp = 0
    # pred_t: pred_t = tp + fp  （tp：真正例，即预测为正且真值为正；fp：假正例，即预测为正但真值为反）
    # 这里 val 时候是指全部预测的实体（无论对错）
    pred_t = 0

    # real_t: real_t = tp + fn  （tp：真正例，即预测为正且真值为正；fn: 假反例，即预测为反但真值为正）
    # 这里 val 时候是指全部人工标注的实体
    real_t = 0

    # 验证
    model.eval()
    with torch.no_grad():
        for iterator, (x_sentences_chars, x_sentences_words, x_sentence_lens, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            x_sentences_chars = x_sentences_chars.to(device=DEVICE)
            x_sentences_words = x_sentences_words.to(device=DEVICE)
            x_sentence_lens = x_sentence_lens.to(device=DEVICE)
            y = y.to(device=DEVICE)

            pred_y = model.predict(x_sentences_chars, x_sentences_words, x_sentence_lens)

            list_list_str_pred_y = LABEL.label_indices2labels(LABEL.to_list(x_sentence_lens, pred_y))
            list_list_str_y = LABEL.label_indices2labels(LABEL.to_list(x_sentence_lens, y))

            for list_str_pred_y, list_str_y in zip(list_list_str_pred_y, list_list_str_y):  # list_str_pred_y, list_str_y 表示某个句子的所有标签
                # set 便于统计
                pre_entities = set(val_utils.bioes_tag2spans(list_str_pred_y))
                entities = set(val_utils.bioes_tag2spans(list_str_y))

                tp += len(pre_entities & entities)  # 交集
                pred_t += len(pre_entities)
                real_t += len(entities)

            if verbose:
                restrict_len = 5
                for i, (list_str_pred_y, list_str_y) in enumerate(zip(list_list_str_pred_y[:restrict_len], list_list_str_y[:restrict_len])):  # list_str_pred_y, list_str_y 表示某个句子的所有标签
                    pre_entities = val_utils.bioes_tag2spans(list_str_pred_y)
                    entities = val_utils.bioes_tag2spans(list_str_y)
                    sentences = WORD_VOCAB.word_indices2words(WORD_VOCAB.to_list(x_sentence_lens[i: i+1], x_sentences_words[i: i+1]))
                    print('---->', sentences, '||||', entities, '||||', pre_entities)

    return val_utils.get_f1_score(tp=tp, pred_t=pred_t, real_t=real_t)


def train():
    # 读取数据
    data_path = 'data'
    # 按照 CNN-LSTM-CRF 那篇论文区分的测试集、训练集等
    # list_list_list_int_sentences_chars(type 为 List[List[List[int]]]): List[int] 为一个 word，List[List[int]] 为一个句子；字符级别的 index
    # list_list_int_sentences_words(type 为 List[List[int]]): int 为一个 word，List[int] 为一个句子；word 级别的 index
    # list_list_int_sentences_labels(type 为 List[List[str]]): int 为一个 word 的 label，List[int] 为一个句子的所有 label；每个单词对应一个 label
    x_val_list_list_list_int_sentences_chars, x_val_list_list_int_sentences_words, y_val_list_list_str_sentences_labels = file_handle.load_pickle(f'{data_path}/eng_bioes_testa.pkl')
    x_train_list_list_list_int_sentences_chars, x_train_list_list_int_sentences_words, y_train_list_list_str_sentences_labels = file_handle.load_pickle(f'{data_path}/eng_bioes_train.pkl')

    # 处理数据(补上 0)
    x_val_sentences_chars = CHAR_VOCAB.to_tensor(x_val_list_list_list_int_sentences_chars, device=None)  # shape: (N, max_sentence_len1, max_word_len1)
    x_val_sentences_words = WORD_VOCAB.to_tensor(x_val_list_list_int_sentences_words, device=None)  # shape: (N, max_sentence_len1)
    x_val_sentence_lens = torch.tensor([len(sentence) for sentence in x_val_list_list_list_int_sentences_chars])  # shape: (N,)
    y_val_labels = LABEL.to_tensor(y_val_list_list_str_sentences_labels, device=None)  # shape: (N, max_sentence_len1)

    x_train_sentences_chars = CHAR_VOCAB.to_tensor(x_train_list_list_list_int_sentences_chars, device=None)  # shape: (N, max_sentence_len2, max_word_len2)
    x_train_sentences_words = WORD_VOCAB.to_tensor(x_train_list_list_int_sentences_words, device=None)  # shape: (N, max_sentence_len2)
    x_train_sentence_lens = torch.tensor([len(sentence) for sentence in x_train_list_list_list_int_sentences_chars])  # shape: (N,)
    y_train_labels = LABEL.to_tensor(y_train_list_list_str_sentences_labels, device=None)  # shape: (N, max_sentence_len1)

    print(f'{x_val_sentences_chars.shape=}, {x_val_sentences_words.shape=}, {y_val_labels.shape=}, {torch.max(x_val_sentence_lens)=}')
    print(f'{x_train_sentences_chars.shape=}, {x_train_sentences_words.shape=}, {y_train_labels.shape=}, {torch.max(x_train_sentence_lens)=}')

    # 装载到 DataLoader
    train_dataset = MyDataset(x_sentences_chars=x_train_sentences_chars, x_sentences_words=x_train_sentences_words, x_sentences_lens=x_train_sentence_lens, y=y_train_labels)
    val_dataset = MyDataset(x_sentences_chars=x_val_sentences_chars, x_sentences_words=x_val_sentences_words, x_sentences_lens=x_val_sentence_lens, y=y_val_labels)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=40)  # 验证集的 batch_size 无所谓

    model = ner.NER(e_char=E_CHAR, e_word=E_WORD, cnn_window_size=CNN_WINDOW_SIZE, cnn_filter_num=CNN_FILTER_NUM, cnn_padding=CNN_PADDING,
                    dropout_p=0.5, hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS, vocab=CHAR_VOCAB, label=LABEL, word_embedding=EMBEDDING, use_crf=USE_CRF)
    model.to(device=DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)  # 见论文 3.2 Optimization Algorithm
    best_f1_score = 0.0

    for epoch in range(EPOCH):
        # 训练
        model.train()
        total_loss = 0.0

        # the learning rate is updated on each epoch of training as ηt = η0/(1 + ρt), with decay rate ρ = 0.05 and t is the number of epoch completed.
        learning_rate = LEARNING_RATE / (1 + DECAY_RATE * epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        for iterator, (x_sentences_chars, x_sentences_words, x_sentence_lens, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            optimizer.zero_grad()  # 梯度缓存清零
            x_sentences_chars = x_sentences_chars.to(device=DEVICE)
            x_sentences_words = x_sentences_words.to(device=DEVICE)
            x_sentence_lens = x_sentence_lens.to(device=DEVICE)
            y = y.to(device=DEVICE)

            loss = model.forward(x_sentences_chars, x_sentences_words, x_sentence_lens, y)  # 标量
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)  # To reduce the effects of “gradient exploding”, we use a gradient clipping of 5.0
            optimizer.step()

            total_loss += loss.item()

            if iterator % 150 == 0 and iterator:
                tqdm.write(f'Round: {epoch=} {iterator=} , Training average loss: {total_loss / iterator}.')

        cur_f1_score, cur_scores = evaluate(model, val_dataloader)
        tqdm.write(f'round: {epoch=}, {cur_scores}')
        if cur_f1_score > best_f1_score:
            best_f1_score = cur_f1_score
            torch.save(model.state_dict(), f'saved_model/crf_epoch_{epoch}_f1_{cur_f1_score:.5%}.pt')


if __name__ == '__main__':
    train()

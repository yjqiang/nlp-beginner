import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from vocab import Vocab
import data_handle
from dataset import MyDataset
from utils import utils
from modules.cnn import Model


USE_GLOVE = True
PATH = 'saved_model/cnn_with_glove' if USE_GLOVE else 'saved_model/cnn_no_glove'

EMBEDDING_SIZE = 100
LIST_FILTER_NUMS = [100, 100, 100]
LIST_WINDOW_SIZES = [3, 4, 5]
MAX_SENTENCE_LENGTH = 48
DROPOUT_P = 0.5
CLASS_NUM = 5
LIST_PADDINGS = None

BATCH_SIZE = 50
EPOCH = 15
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 看看是否有预训练的 glove
if USE_GLOVE:
    print('发现预训练的词向量，加载...')
    list_embedding, word2id = data_handle.read_glove("data/glove.6B.100d.txt")
    # glove 缺失 pad 和 unk，手动帮助生成
    pad_index = len(list_embedding)
    list_embedding.append(torch.zeros(EMBEDDING_SIZE, dtype=torch.float).tolist())
    word2id['<pad>'] = pad_index
    unk_index = len(list_embedding)
    list_embedding.append(torch.randn(EMBEDDING_SIZE).tolist())
    word2id['<unk>'] = unk_index

    VOCAB = Vocab(word2id)
    tensor_embedding = torch.tensor(list_embedding).to(DEVICE)
    assert tensor_embedding.shape[1] == EMBEDDING_SIZE  # 确保维度一致
    EMBEDDING = nn.Embedding.from_pretrained(tensor_embedding, freeze=False, padding_idx=VOCAB.pad_index)  # the tensor does get updated in the learning process!! 微调（其实就是个初始化）
else:
    print('未发现预训练的词向量')
    VOCAB = Vocab.load_json('vocab.json')
    EMBEDDING = None


def evaluate(model: nn.Module, dataloader: DataLoader, dataset: MyDataset) -> float:
    sum_num = len(dataset)  # 多少个 iterator
    corr_num = 0
    # 验证
    model.eval()
    with torch.no_grad():
        for x, y, _ in tqdm(dataloader, total=len(dataloader)):
            x = x.to(DEVICE)
            y = y.to(DEVICE)  # shape: (batch_size, )   y[i] 是 x[i] 的分类真值，且 0 <= y[i] < CLASS_NUM
            scores = model.get_scores(x)  # shape: (batch_size, CLASS_NUM)
            corr_num += torch.eq(scores.argmax(dim=1), y).sum().item()
    cur_accuracy = corr_num / sum_num
    return cur_accuracy


def train():
    # 读取数据
    x_train_orig, y_train_orig = data_handle.read_tsv_data0('data/train_split.tsv')
    x_val_orig, y_val_orig = data_handle.read_tsv_data0('data/val_split.tsv')

    x_train_sentences = utils.split_sentences(x_train_orig)
    x_train, seq_lengths_train = VOCAB.to_input_tensor(x_train_sentences, MAX_SENTENCE_LENGTH, device=None)  # 一口气全丢到 cuda 里面，你不怕炸么
    y_train = torch.tensor(y_train_orig)
    x_val_sentences = utils.split_sentences(x_val_orig)
    x_val, seq_lengths_val = VOCAB.to_input_tensor(x_val_sentences, MAX_SENTENCE_LENGTH, device=None)  # 一口气全丢到 cuda 里面，你不怕炸么
    y_val = torch.tensor(y_val_orig)
    x_val, y_val, seq_lengths_val = x_val[:x_val.shape[0] // 2], y_val[:y_val.shape[0] // 2], seq_lengths_val[:len(seq_lengths_val) // 2]  # task01 数据基础上，把 val 再劈一个 test，见 cnn_load_and_test.py
    print(f'{x_train.shape=}, {y_train.shape=}')
    print(f'{x_val.shape=}, {y_val.shape=}')

    # 装载到 DataLoader
    train_dataset = MyDataset(x=x_train, y=y_train, seq_lengths=seq_lengths_train)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    val_dataset = MyDataset(x=x_val, y=y_val, seq_lengths=seq_lengths_val)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # 训练
    model = Model(vocab=VOCAB, embedding_size=EMBEDDING_SIZE, list_filter_nums=LIST_FILTER_NUMS, list_window_sizes=LIST_WINDOW_SIZES,
                  max_sentence_length=MAX_SENTENCE_LENGTH, dropout_p=DROPOUT_P, class_num=CLASS_NUM, list_paddings=LIST_PADDINGS, embedding=EMBEDDING)
    model.to(device=DEVICE)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    best_accuracy = 0.0
    for epoch in range(EPOCH):
        # 训练
        model.train()
        total_loss = 0.0
        for iterator, (x, y, _) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            optimizer.zero_grad()  # 梯度缓存清零
            x = x.to(device=DEVICE)
            y = y.to(device=DEVICE)  # shape: (batch_size, )   y[i] 是 x[i] 的分类真值，且 0 <= y[i] < CLASS_NUM
            loss = model(x, y)  # 标量
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if iterator % 100 == 0 and iterator:
                tqdm.write(f'Round: {epoch=} {iterator=} , Training average loss: {total_loss / iterator}.')

        cur_accuracy = evaluate(model, val_dataloader, val_dataset)
        tqdm.write(f'Round: {epoch=}, Validation accuracy:{cur_accuracy:.5%}')
        if best_accuracy < cur_accuracy:
            best_accuracy = cur_accuracy
            torch.save(model.state_dict(), f'{PATH}/0epoch_{epoch}_accuracy_{cur_accuracy:.5%}.pt')


if __name__ == '__main__':
    train()

from pathlib import Path
import sys

from sklearn.model_selection import train_test_split

import data_handle
import features
import models


NUM_CLASSES = 5
BATCH_SIZE = 64
EPOCH = 500
LEARNING_RATE = 1e-3

# 如果没有分割好的数据，那么就进行分割（否则每次运行，x_train 数据不一样，词袋模型的生成词典大小也不一样，那些全连接层参数大小不固定）
if not (Path("data/train_split.tsv").is_file() and Path("data/val_split.tsv").is_file()):
    sentences, labels = data_handle.read_tsv_data0('data/train.tsv')  # 读取数据
    x_train_orig, x_val_orig, y_train, y_val = train_test_split(sentences, labels, test_size=0.25)  # 按照 75: 25 比例划分 训练集和验证集；结果的 type 不变，仍为 pd.Series
    data_handle.save_tsv_data([x_train_orig, y_train], 'data/train_split.tsv')
    data_handle.save_tsv_data([x_val_orig, y_val], 'data/val_split.tsv')
else:
    x_train_orig, y_train = data_handle.read_tsv_data0('data/train_split.tsv')
    x_val_orig, y_val = data_handle.read_tsv_data0('data/val_split.tsv')

n_gram = 1

# 特征提取
# n-gram 加上 词袋模型
bog = features.BagOfWordFeature(n_gram=n_gram)
x_train = bog.fit(x_train_orig)
x_val = bog.transform(x_val_orig)
print(f'训练集：{x_train.shape=}, {y_train.shape=}')
print(f'验证集：{x_val.shape=}, {y_val.shape=}')


model = models.Model(input_features=x_train.shape[1], out_features=NUM_CLASSES)
model.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
            batch_size=BATCH_SIZE, epoch_num=EPOCH, learning_rate=LEARNING_RATE, reg=0)



# (117045, 15227) (39015, 15227)
# 0.5314622581058567

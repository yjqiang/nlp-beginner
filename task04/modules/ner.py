import torch
from torch import nn

from .lstm import LSTM
from .char_cnn import CharCNNEmbedding
from .crf import CRFDecoder
from .softmax import SoftmaxDecoder
from vocab import CharVocab
from label import Label


class NER(nn.Module):
    def __init__(self, e_char: int, word_embedding: nn.Embedding, e_word: int, cnn_window_size: int, cnn_filter_num: int, cnn_padding: int,
                 dropout_p: float, hidden_size: int, num_layers: int, vocab: CharVocab, label: Label, use_crf: bool = True):
        """
        :param e_char: char 的向量维数
        :param word_embedding: 预训练的词向量
        :param e_word: Embedding size (dimensionality) for the output word  词向量的维数（注意是“词”向量！！）
        :param dropout_p: p
        :param cnn_window_size: cnn filter 的窗口大小
        :param cnn_filter_num: cnn filter 数目，也是 char cnn 的输出维度（每个 word 的对应向量维数）
        :param cnn_padding: cnn padding
        :param hidden_size: (LSTM) 注意由于是 双向 LSTM，所以要除以 2
        :param num_layers: (LSTM) num_layers
        :param vocab: CharVocab object. See vocab.py for documentation. 全局共享即可
        :param label: Label object. See label.py for documentation. 全局共享即可
        """
        super().__init__()

        # (N, max_sentence_len, max_word_len) -> (N, max_sentence_len, filter_num)
        self.char_nn_part = CharCNNEmbedding(e_char=e_char, filter_num=cnn_filter_num, window_size=cnn_window_size, dropout_p=dropout_p, vocab=vocab, padding=cnn_padding)
        # (N, max_sentence_len) -> (N, max_sentence_len, e_word)
        self.word_embedding = word_embedding

        assert not hidden_size % 2
        # Figure 3: Then the character representation vector is concatenated with the word embedding before feeding into the BLSTM network.
        self.lstm_part = LSTM(input_size=e_word+cnn_filter_num, hidden_size=hidden_size//2, num_layers=num_layers)

        self.dropout_part = nn.Dropout(p=dropout_p)

        self.decoder = CRFDecoder(hidden_size=hidden_size, label=label) if use_crf else SoftmaxDecoder(hidden_size=hidden_size, label=label)

    def cal(self, x_sentences_chars: torch.Tensor, x_sentences_words: torch.Tensor, x_sentence_lens: torch.Tensor) -> torch.Tensor:
        """
        forward 和 predict 有很多公共事务，合并
        :param x_sentences_chars: shape: (N, max_sentence_len, max_word_len) 每个元素都是一个 char 的 index（所有单词最长为 max_word_len）
        :param x_sentences_words: shape: (N, max_sentence_len) 每个元素都是一个 word 的 index（所有句子最长为 max_sentence_len）
        :param x_sentence_lens: 每句话的实际长度；shape: (N,)

        :return: lstm 的结果；shape: (N, max_sentence_len, hidden_size)
        """
        x_char_cn = self.char_nn_part.forward(x_sentences_chars)  # shape: (N, max_sentence_len, filter_num)
        x_word_embedding = self.word_embedding(x_sentences_words)  # shape: (N, max_sentence_len, e_word)

        # Figure 3: Then the character representation vector is concatenated with the word embedding before feeding into the BLSTM network.
        x = torch.cat([x_char_cn, x_word_embedding], -1)  # shape: (N, max_sentence_len, filter_num+e_word)

        # As shown in Figure 3, dropout layers are applied on both the input and output vectors of BLSTM.
        x = self.dropout_part(x)
        lstm_result = self.lstm_part.forward(x, x_sentence_lens)  # shape: (N, max_sentence_len, hidden_size)
        lstm_result = self.dropout_part(lstm_result)

        return lstm_result

    def forward(self, x_sentences_chars: torch.Tensor, x_sentences_words: torch.Tensor, x_sentence_lens: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        获取 loss，为了梯度下降
        :param x_sentences_chars: shape: (N, max_sentence_len, max_word_len) 每个元素都是一个 char 的 index（所有单词最长为 max_word_len）
        :param x_sentences_words: shape: (N, max_sentence_len) 每个元素都是一个 word 的 index（所有句子最长为 max_sentence_len）
        :param x_sentence_lens: 每句话的实际长度；shape: (N,)
        :param y: shape: (N, max_sentence_len)  y[i] 是 x[i][j]（第 i 句的第 j 个 word） 的分类真值  (不含 START END 即 仅含实际句子的所有标签)

        :return: 标量
        """
        lstm_result = self.cal(x_sentences_chars=x_sentences_chars, x_sentences_words=x_sentences_words, x_sentence_lens=x_sentence_lens)  # lstm 的结果；shape: (N, max_sentence_len, hidden_size)

        return self.decoder.forward(lstm_result=lstm_result, x_sentence_lens=x_sentence_lens, y=y)

    def predict(self, x_sentences_chars: torch.Tensor, x_sentences_words: torch.Tensor, x_sentence_lens: torch.Tensor) -> torch.Tensor:
        """
        预测
        :param x_sentences_chars: shape: (N, max_sentence_len, max_word_len) 每个元素都是一个 char 的 index（所有单词最长为 max_word_len）
        :param x_sentences_words: shape: (N, max_sentence_len) 每个元素都是一个 word 的 index（所有句子最长为 max_sentence_len）
        :param x_sentence_lens: 每句话的实际长度；shape: (N,)

        :return: shape: (N, max_sentence_len)  预测的标注
        """
        lstm_result = self.cal(x_sentences_chars=x_sentences_chars, x_sentences_words=x_sentences_words, x_sentence_lens=x_sentence_lens)  # lstm 的结果；shape: (N, max_sentence_len, hidden_size)

        return self.decoder.predict(lstm_result=lstm_result, x_sentence_lens=x_sentence_lens)

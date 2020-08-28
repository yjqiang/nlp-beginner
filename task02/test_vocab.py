from vocab import Vocab

vocab = Vocab.load('vocab.json')
for (word0, index0), (index1, word1) in zip(vocab.word2id.items(), vocab.id2word.items()):
    assert word0 == word1
    assert index0 == index1

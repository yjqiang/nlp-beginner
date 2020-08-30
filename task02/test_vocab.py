from vocab import Vocab

vocab = Vocab.load_json('vocab.json')
for word, index in vocab.word2id.items():
    print(word, index)

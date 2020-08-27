import pandas as pd

df = pd.read_csv('data/val_split.tsv', delimiter='\t')

print(df.info())
print(df['Phrase'][:5].tolist())

a = df['Phrase']
b = df['Phrase']
print(isinstance(a, pd.Series))
print(a[:5].tolist())
print(a)




print(df.Sentiment.value_counts()/df.Sentiment.count())

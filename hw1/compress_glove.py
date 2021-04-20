import numpy as np
import nltk
import psutil
from sklearn.feature_extraction.text import CountVectorizer
from data_io import *

if __name__ == '__main__':
  print("Loading glove.6B")
  with open("data/glove.6B/glove.6B.50d.txt", "r") as f:
    word2vec = {}
    for line in f.readlines():
      linesplit = line.split()
      word2vec[linesplit[0]] = np.array([float(x) for x in linesplit[1:]])
  d = len(next(iter(word2vec.values())))
  print(f'dim = {d}')
  print(psutil.virtual_memory())

  print("Compressing...")
  labeled = load_data()
  unlabeled = load_unlabeled_data()
  data = labeled.data + unlabeled.data
  count_vect = CountVectorizer(tokenizer=nltk.word_tokenize)
  count_vect.fit(data)
  vocabulary = count_vect.vocabulary_.keys()
  compressed_w2v = {}
  for w in vocabulary:
    if w in word2vec:
      compressed_w2v[w] = word2vec[w]
    else:
      compressed_w2v[w] = np.zeros(d)
  del word2vec
  print(psutil.virtual_memory())

  print("Saving compressed word2vec into data/compressed_glove.6B.txt")
  with open('data/compressed_glove.6B.txt', 'w') as f:
    for w, vec in compressed_w2v.items():
      f.write(f"{w} {' '.join(map(str, vec.tolist()))}")
      f.write('\n')

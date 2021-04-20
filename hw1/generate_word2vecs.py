import gensim.models
import nltk
from data_io import *

if __name__ == '__main__':
  labeled = load_data()
  unlabeled = load_unlabeled_data()
  data = labeled.data + unlabeled.data
  sentences = [nltk.word_tokenize(doc.lower()) for doc in data]

  window_sizes = [5, 10, 15, 20]
  dimensions = [100, 150, 200]

  for window_size in window_sizes:
    for dim in dimensions:
      print(f'Generating for window={window_size} and dimension={dim}...', end='')
      w2v = gensim.models.Word2Vec(
          sentences=sentences,
          vector_size=dim,
          window=window_size
      )
      print('done')

      # Save
      outfile = f'data/word2vecs/labeled+unlabeled_{window_size}_{dim}.txt'
      print(f"Saving word2vec embeddings into {outfile}")
      with open(f'{outfile}', 'w') as f:
        for w in w2v.wv.index_to_key:
          f.write(f"{w} {' '.join(map(str, w2v.wv[w].tolist()))}\n")

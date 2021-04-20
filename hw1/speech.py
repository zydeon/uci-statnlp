#!/bin/python

import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator
import pandas as pd
import nltk
import time
import os
import sys
import numpy as np
from data_io import *


def load_word2vecs(windows=(5, 10, 15, 20), dimensions=([100, 150, 200])):
  word2vecs = {}
  for window in windows:
    for dim in dimensions:
      print(f"Loading w2v window={window} dimension={dim}...", end='')
      with open(f'data/word2vecs/labeled+unlabeled_{window}_{dim}.txt', "r") as f:
        word2vecs[f'w{window}_d{dim}'] = {}
        for line in f.readlines():
          linesplit = line.split()
          word2vecs[f'w{window}_d{dim}'][linesplit[0]] = np.array([float(x) for x in linesplit[1:]])
      print("done")
  return word2vecs

# if __name__ == '__main__':
WORD2VECS = load_word2vecs()


class W2VAggVectorizer(BaseEstimator):

  def __init__(self, kind=None, agg=None):  # window=None, dim=None):
    self.kind = kind
    self.agg = agg

  def set_params(self, **params):
    self.agg = self._agg_vectorizer_factory(params['kind'])
    super().set_params(**params)

  def _agg_vectorizer_factory(self, kind):
    if kind == 'avgw2v_tfidf':
      return W2VAggVectorizerAvgTfidf()
    elif kind == 'avgw2v':
      return W2VAggVectorizerAvg()
    elif kind == 'sumw2v':
      return W2VAggVectorizerSum()
    elif kind == 'minw2v':
      return W2VAggVectorizerMin()
    elif kind == 'maxw2v':
      return W2VAggVectorizerMax()
    elif kind == 'minmaxw2v':
      return W2VAggVectorizerMinMax()

  def fit(self, X, y=None):
    return self.agg.fit(X, y)

  def transform(self, X):
    return self.agg.transform(X)


class W2VAggVectorizerAvgTfidf(BaseEstimator):
  def __init__(self, tfidf=TfidfVectorizer(tokenizer=nltk.word_tokenize), window=None, dim=None):
    self.tfidf = tfidf
    self.dim = dim
    self.window = window

  def fit(self, X, y=None):
    self.tfidf.fit(X, y)
    vocabulary_items = sorted(self.tfidf.vocabulary_.items(), key=lambda pair: pair[1])  # Sort by index
    vocabulary = [w for w, ix in vocabulary_items]
    self.w2v = np.array([WORD2VECS[f'w{self.window}_d{self.dim}'].get(w, np.zeros(self.dim)) for w in vocabulary])
    return self

  def transform(self, X):
    # Compute TF-IDF matrix.
    tfidf_values = self.tfidf.transform(X).toarray()
    tfidf_values = normalize(tfidf_values, norm='l1')  # Rows sum to 1 for taking average.

    # Compute weighted average of word2vec vectors, using tf-idf weights
    return np.dot(tfidf_values, self.w2v)

  def fit_transform(self, X, y=None):
    self.fit(X, y)
    return self.transform(X)


class W2VAggVectorizerAvg(BaseEstimator):
  def __init__(self, bow=CountVectorizer(tokenizer=nltk.word_tokenize), window=None, dim=None):
    self.bow = bow
    self.dim = dim
    self.window = window

  def fit(self, X, y=None):
    self.bow.fit(X, y)
    vocabulary_items = sorted(self.bow.vocabulary_.items(), key=lambda pair: pair[1])  # Sort by index
    vocabulary = [w for w, ix in vocabulary_items]
    self.w2v = np.array([WORD2VECS[f'w{self.window}_d{self.dim}'].get(w, np.zeros(self.dim)) for w in vocabulary])
    return self

  def transform(self, X):
    # Compute bow matrix.
    bow_values = self.bow.transform(X).toarray()
    bow_values = normalize(bow_values, norm='l1')  # Rows sum to 1 for taking average.

    # Compute weighted average of word2vec vectors
    return np.dot(bow_values, self.w2v)


class W2VAggVectorizerSum(BaseEstimator):
  def __init__(self, bow=CountVectorizer(tokenizer=nltk.word_tokenize), window=None, dim=None):
    self.bow = bow
    self.dim = dim
    self.window = window

  def fit(self, X, y=None):
    self.bow.fit(X, y)
    vocabulary_items = sorted(self.bow.vocabulary_.items(), key=lambda pair: pair[1])  # Sort by index
    vocabulary = [w for w, ix in vocabulary_items]
    self.w2v = np.array([WORD2VECS[f'w{self.window}_d{self.dim}'].get(w, np.zeros(self.dim)) for w in vocabulary])
    return self

  def transform(self, X):
    # Compute bow matrix.
    bow_values = self.bow.transform(X).toarray()

    # Compute weighted average of word2vec vectors
    return np.dot(bow_values, self.w2v)


class W2VAggVectorizerMin(BaseEstimator):
  def __init__(self, bow=CountVectorizer(tokenizer=nltk.word_tokenize, binary=True), window=None, dim=None):
    self.bow = bow
    self.dim = dim
    self.window = window

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return np.array([
        np.min([
            WORD2VECS[f'w{self.window}_d{self.dim}'].get(w, np.zeros(self.dim))
            for w in nltk.word_tokenize(doc)
        ], axis=0)
        for doc in X
    ])


class W2VAggVectorizerMax(BaseEstimator):
  def __init__(self, bow=CountVectorizer(tokenizer=nltk.word_tokenize, binary=True), window=None, dim=None):
    self.bow = bow
    self.dim = dim
    self.window = window

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return np.array([
        np.max([
            WORD2VECS[f'w{self.window}_d{self.dim}'].get(w, np.zeros(self.dim))
            for w in nltk.word_tokenize(doc)
        ], axis=0)
        for doc in X
    ])


class W2VAggVectorizerMinMax(BaseEstimator):
  def __init__(self, bow=CountVectorizer(tokenizer=nltk.word_tokenize, binary=True), window=None, dim=None):
    self.bow = bow
    self.dim = dim
    self.window = window

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return np.concatenate(
        (
            np.array([
                np.min([
                    WORD2VECS[f'w{self.window}_d{self.dim}'].get(w, np.zeros(self.dim))
                    for w in nltk.word_tokenize(doc)
                ], axis=0)
                for doc in X
            ]),
            np.array([
                np.max([
                    WORD2VECS[f'w{self.window}_d{self.dim}'].get(w, np.zeros(self.dim))
                    for w in nltk.word_tokenize(doc)
                ], axis=0)
                for doc in X
            ])
        ), axis=1
    )


def get_features_transformer(features_type):
  if features_type == 'bow':
    return CountVectorizer()
  elif features_type == 'tfidf':
    return TfidfVectorizer(tokenizer=nltk.word_tokenize)
  elif features_type == 'tfidf_stopwords':
    return TfidfVectorizer(tokenizer=nltk.word_tokenize,
                           stop_words=nltk.corpus.stopwords.words("english"))
  elif features_type == 'bow+tfidf':
    return FeatureUnion([
        ('a', TfidfVectorizer()),
        ('b', CountVectorizer())
    ])
  elif features_type == 'w2vagg':
    return TfidfEmbeddingVectorizer()
  elif features_type == 'tfidf+w2vagg':
    return FeatureUnion([
        ('tfidf', TfidfVectorizer(tokenizer=nltk.word_tokenize)),
        ('w2vagg', W2VAggVectorizer())
    ])
  else:
    raise Exception("Type of features not handled")


def summary_results(df_results):
  main_cols = ['params', 'mean_train_score', 'mean_test_score', 'std_train_score', 'std_test_score']
  return df_results[main_cols]


if __name__ == "__main__":
  # Get param grid configs.
  param_grid = json.load(sys.stdin)

  # Load data.
  print("Reading data")
  labeled = load_data()

  # Get features_type
  features_type = param_grid.pop('features_type')

  # Create output folder.
  print("Creating results folder")
  ts = time.strftime('%Y-%m-%d_%Hh%Mm%S', time.localtime())
  out_folder = f'data/results/{ts}_{features_type}'
  os.mkdir(out_folder)

  # Save hyperparams ranges.
  print(f"Saving param_grid to {out_folder}/param_grid.json")
  with open(f'{out_folder}/param_grid.json', 'w') as f:
    json.dump(param_grid, f, indent=2)

  # Perform grid search.
  print("Doing grid search on params")
  t0 = time.time()
  pipe = Pipeline([
      (f'{features_type}', get_features_transformer(features_type)),
      ('lr', LogisticRegression())
  ])

  gs = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, verbose=4, return_train_score=True)
  gs.fit(labeled.data, labeled.y)
  print("[Grid search] Done in %0.3fs\n" % (time.time() - t0))

  # Save results
  print(f"Saving results to {out_folder}...", end='')
  df_results = pd.DataFrame(gs.cv_results_)
  df_results.to_csv(f'{out_folder}/results_verbose.csv')
  summary_results(df_results).to_csv(f'{out_folder}/results.csv')
  print(' Done')

  # Save best
  print(gs.best_score_)
  print(gs.best_params_)
  with open(f'{out_folder}/best_params.json', 'w') as f:
    json.dump({'mean_test_score': gs.best_score_, **gs.best_params_}, f)

  # Predict unlabeled with best parameters setting.
  print("Predicting unlabeled data with best params...")
  yp = gs.predict(np.array(load_unlabeled_data().data))
  write_pred_kaggle_file(yp, f'{out_folder}/sup.csv', labeled.le)
  print("[done]")

  # Save best weights
  # print(gs.best_estimator_['lr'].coef_)

#!/bin/python

import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import pandas as pd
import nltk
import time
import os
import sys
from sklearn.base import BaseEstimator
import numpy as np
from data_io import *

A = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
B = 'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB'


class Tfidf(BaseEstimator):
  def __init__(self, max_df=3.0, ngram_range=(1, 1), a=True):
    self.tfidf = TfidfVectorizer(max_df=max_df, ngram_range=ngram_range)
    self.max_df = max_df
    self.ngram_range = ngram_range
    self.a = a

  # def set_params(self, **params):
  #   self.tfidf.set_params(**params)

  def fit(self, X, y):
    if self.a:
      print(A)
    else:
      print(B)
    self.tfidf.fit(X, y)
    return self

  def transform(self, X):
    return self.tfidf.transform(X)


if __name__ == "__main__":
  labeled = load_data()
  param_grid = {
      'tfidf__max_df': [1.0],
      'tfidf__ngram_range': [[1, 1]],
      "lr__class_weight": [
        None
      ],
      'tfidf__a': [True, False]
  }
  pipe = Pipeline([
      ('tfidf', Tfidf()),
      ('lr', LogisticRegression(n_jobs=-1))
  ])
  gs = GridSearchCV(pipe, param_grid=param_grid, n_jobs=1, verbose=4, cv=5)
  gs.fit(np.array(labeled.data), labeled.y)
  df_results = pd.DataFrame(gs.cv_results_)
  print(df_results)


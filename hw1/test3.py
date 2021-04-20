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


class B(BaseEstimator):
  def __init__(self, b0=None):
    self.b0 = b0

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return [self.b0]


class C(BaseEstimator):
  def __init__(self, c0=None):
    self.c0 = c0

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return [self.c0]


class A(BaseEstimator):
  def __init__(self, b=B()):
    self.b = b

  def fit(self, X, y=None):
    self.b.fit(X, y)
    return self

  def transform(self, X):
    return [self.b.transform(X)]


class Estimator:
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return X

  def score(self, X, y, sample_weight=None):
    return 0

  def predict(self, X):
    self.fit(X)
    return self.transform(X)

  # def fit_transform(self, X, y=None):
  #   self.fit(X, y)
  #   return self.transform(X)


if __name__ == "__main__":
  param_grid = {
      'a__b__b0': [2, 3]
  }
  pipe = Pipeline([
      ('a', A()),
      ('estimator', Estimator())
  ])
  print(A().get_params().keys())
  gs = GridSearchCV(pipe, param_grid=param_grid, n_jobs=1, verbose=4, cv=2)
  X = [0, 0]
  gs.fit(X)
  print(gs.predict(X))
  print(gs.best_estimator_['estimator'].coef_)
  # print(gs.predict(X))

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


class A(BaseEstimator):
  def __init__(self, a0=None, a1=None):
    self.a0 = a0
    self.a1 = a1

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return [self.a0, self.a1]


class B(BaseEstimator):
  def __init__(self, b0=None, b1=None):
    self.b0 = b0
    self.b1 = b1

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return [self.b0, self.b1]


class Estimator:
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return X

  def score(self, X, y, sample_weight=None):
    return 0

  def predict(self, X)

  # def fit_transform(self, X, y=None):
  #   self.fit(X, y)
  #   return self.transform(X)


if __name__ == "__main__":
  param_grid = {
      'features__a__a0': [2],
      'features__a__a1': [5],
      'features__b__b1': [0],
      'features__b__b1': [0]
  }
  pipe = Pipeline([
      ('features', FeatureUnion([
          ('a', A()),
          ('b', B())
      ])),
      ('estimator', Estimator())
  ])
  gs = GridSearchCV(pipe, param_grid=param_grid, n_jobs=1, verbose=4, cv=2)
  X = [0, 0]
  gs.fit(X)
  gs.predict(X)
  # print(gs.predict(X))

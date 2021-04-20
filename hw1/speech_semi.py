#!/bin/python

import pandas as pd
import nltk
import time
import os
import numpy as np
import sys
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture

from data_io import *
from speech import *


def load_unlabeled_labeled_data(tarfname=''):
  print("-- unlabeled labeled data")
  data, fnames, labels = read_tsv("data/results_unlabeled_labels/self_labeled.tsv")
  print(len(data))

  speech = Data()
  speech.data = data
  speech.fnames = fnames
  speech.labels = labels

  # Labels
  speech.le = preprocessing.LabelEncoder()
  speech.le.fit(speech.labels)
  speech.target_labels = speech.le.classes_
  speech.y = speech.le.transform(speech.labels)
  return speech


if __name__ == "__main__":
  print("Reading data")
  unlabeled_labeled = load_unlabeled_labeled_data()

  # Create output folder.
  print("Creating results folder")
  ts = time.strftime('%Y-%m-%d_%Hh%Mm%S', time.localtime())
  out_folder = f'data/results_semi/{ts}'
  os.mkdir(out_folder)

  param_grid = {
      "lr__C": [80.0],
      "lr__class_weight": ["balanced"],
      "lr__max_iter": [200],
      "lr__n_jobs": [-1],
      "lr__penalty": ["l2"],
      "lr__tol": [0.0001],
      "tfidf__max_df": [0.9],
      "tfidf__ngram_range": [[1, 2]]
  }

  # Save hyperparams ranges.
  print(f"Saving param_grid to {out_folder}/param_grid.json")
  with open(f'{out_folder}/param_grid.json', 'w') as f:
    json.dump(param_grid, f, indent=2)

  t0 = time.time()
  pipe_train = Pipeline([
      (f'tfidf', get_features_transformer('tfidf')),
      ('lr', LogisticRegression())
  ])

  gs = GridSearchCV(pipe_train, param_grid=param_grid, n_jobs=-1, verbose=4, return_train_score=True)
  gs.fit(unlabeled_labeled.data, unlabeled_labeled.y)
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
  write_pred_kaggle_file(yp, f'{out_folder}/sup.csv', unlabeled_labeled.le)
  print("[done]")

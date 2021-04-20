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

if __name__ == "__main__":
  # Load data.
  print("Reading data")
  labeled = load_data()
  unlabeled = load_unlabeled_data()
  all_data = labeled.data + unlabeled.data

  # Create output folder.
  print("Creating results folder")
  ts = time.strftime('%Y-%m-%d_%Hh%Mm%S', time.localtime())
  out_folder = f'data/results/cluster_{ts}'
  os.mkdir(out_folder)

  embedding_vectorizer = lambda: W2VAggVectorizerAvgTfidf(
      tfidf=TfidfVectorizer(tokenizer=nltk.word_tokenize, max_df=0.8), window=20, dim=200
  )
  t0 = time.time()
  pipe19_gmm = Pipeline([
      (f'avgw2v_tfidf', embedding_vectorizer()),
      ('gmm', GaussianMixture(n_components=19,
                              init_params='random',
                              verbose=10))
  ])
  print("Computing clusters...", end='')
  pipe19_gmm.fit(np.array(all_data))
  print("done")
  print("%0.2fs" % (time.time() - t0))

  # Find mapping from centroids to candidates from proportion of labeled docs in each cluster
  X = pipe19_gmm['avgw2v_tfidf'].transform(np.array(labeled.data))
  membership_probs = pipe19_gmm['gmm'].predict_proba(X)

  # Membership probabilities for each candidate, given a cluster
  df = pd.DataFrame(membership_probs)
  df['y'] = labeled.labels
  candidates_probs = df.groupby('y').sum()
  candidates_probs_norm = normalize(candidates_probs, norm='l1', axis=0)  # so columns sum to 1

  # Determine mapping probabilistically
  mapping = pd.DataFrame([np.random.choice(candidates_probs.index, p=col) for col in candidates_probs_norm.T])

  # Predict centroids for unlabeled data
  centroid_predictions = pipe19_gmm.predict(unlabeled.data)

  # Map into candidate names
  unlabeled_labels = pd.Series(centroid_predictions).map(mapping.squeeze())

  # Save into file
  pd.DataFrame(dict(
      fname=list(map(lambda fname: f'unlabeled/{fname}', unlabeled.fnames)),
      label=unlabeled_labels
  )).to_csv('data/results_unlabeled_labels/self_labeled.tsv', sep='\t', header=False, index=False)

#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import itertools
from math import log
import numpy as np
import sys


def get_item(dic, key):
  try:
    return dic[key]
  except KeyError:
    return 0

class defaultdict(dict):
  """ Better implementation of defaultdict that does not add elements to dictionary
  when simply accessing an unexisting key."""

  def __init__(self, default_factory):
    self.default_factory = default_factory
    super().__init__(self)

  def __getitem__(self, k):
    try:
      return super().__getitem__(k)
    except KeyError:
      return 0


# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

  def xrange(*args, **kwargs):
    return iter(range(*args, **kwargs))

  def unicode(*args, **kwargs):
    return str(*args, **kwargs)


SOS = 'START_OF_SENTENCE'
EOS = 'END_OF_SENTENCE'
UNK = '<UNK>'


class LangModel:
  def fit_corpus(self, corpus):
    """Learn the language model for the whole corpus.

    The corpus consists of a list of sentences."""
    corpus = self.prefit(corpus)
    for s in corpus:
      self.fit_sentence(s)
    self.postfit()
    self.norm()

  def perplexity(self, corpus):
    """Computes the perplexity of the corpus by the model.

    Assumes the model uses an EOS symbol at the end of each sentence.
    """
    numOOV = self.get_num_oov(corpus)
    return pow(2.0, self.entropy(corpus, numOOV))

  def get_num_oov(self, corpus):
    vocab_set = set(self.vocab())
    words_set = set(itertools.chain(*corpus))
    numOOV = len(words_set - vocab_set)
    return numOOV

  def entropy(self, corpus, numOOV):
    num_words = 0.0
    sum_logprob = 0.0
    for s in corpus:
      num_words += len(s) + 1  # for EOS
      sum_logprob += self.logprob_sentence(s, numOOV)
    return -(1.0 / num_words) * (sum_logprob)

  def logprob_sentence(self, sentence, numOOV):
    """ redefined for efficiency. """
    p = 0.0
    for i in xrange(len(sentence)):
      p += self.cond_logprob(sentence[i], sentence[:i], numOOV)
    p += self.cond_logprob(EOS, sentence, numOOV)
    return p

  # required, update the model when a sentence is observed
  def fit_sentence(self, sentence):
    pass

  # optional, if there are any post-training steps (such as normalizing probabilities)
  def norm(self):
    pass

  # Useful to do stats.
  def postfit(self):
    pass

  # Useful to precompute frequencies.
  def prefit(self, corpus):
    return corpus

  # required, return the log2 of the conditional prob of word, given previous words
  def cond_logprob(self, word, previous, numOOV):
    pass

  # required, the list of words the language model suports (including EOS)
  def vocab(self):
    pass


class Unigram(LangModel):
  def __init__(self, unk_prob=0.0001):
    self.model = dict()
    self.lunk_prob = log(unk_prob, 2)

  def postfit(self):
    pass

  def inc_word(self, w):
    if w in self.model:
      self.model[w] += 1.0
    else:
      self.model[w] = 1.0

  def fit_sentence(self, sentence):
    for w in sentence:
      self.inc_word(w)
    self.inc_word(EOS)

  def norm(self):
    """Normalize and convert to log2-probs."""
    tot = 0.0
    for word in self.model:
      tot += self.model[word]
    ltot = log(tot, 2)
    for word in self.model:
      self.model[word] = log(self.model[word], 2) - ltot

  def cond_logprob(self, word, previous, numOOV):
    if word in self.model:
      return self.model[word]
    else:
      return self.lunk_prob - log(numOOV, 2)

  def vocab(self):
    return self.model.keys()


class Ngram(LangModel):
  def __init__(self, n):
    self.n = n

  def postfit(self):
    pass

  def logprob_sentence(self, sentence, numOOV):
    """ redefined for efficiency, reduces memory overhead of sentence slices"""
    p = 0.0
    for i in xrange(len(sentence)):
      p += self.cond_logprob(sentence[i], sentence[i - self.n + 1:i], numOOV)
    p += self.cond_logprob(EOS, sentence[-self.n + 1:], numOOV)
    return p

  def _get_ngrams(sentence, n):
    # Add start tokens, to model distribution for starting words.
    sentence = (n - 1) * [SOS] + sentence + [EOS]
    return zip(*[sentence[i:] for i in range(n)])


class NgramNoUnk(Ngram):
  def __init__(self, n, λ):
    self.n = n
    self.λ = λ  # lambda for laplace smoothing.
    self.vocabulary = set()
    self.ngram_count = {}  # n-grams
    self.context_count = {}  # (n-1)-grams

  def postfit(self):
    # Set vocabulary size.
    self.V = len(self.vocab())

  # required, the list of words the language model suports (including EOS)
  def vocab(self):
    return self.vocabulary

  # required, update the model when a sentence is observed
  def fit_sentence(self, sentence):
    ngrams = Ngram._get_ngrams(sentence, self.n)
    for ngram in ngrams:
      self.vocabulary.add(ngram[-1])
      context = ngram[:-1]
      self.ngram_count[ngram] = self.ngram_count.get(ngram, 0) + 1
      self.context_count[context] = self.context_count.get(context, 0) + 1

  # required, return the log2 of the conditional prob of word, given previous words
  def cond_logprob(self, word, context, numOOV):
    # Add start tokens if not enough context, to model distribution for starting words.
    if len(context) < self.n - 1:
      context = (self.n - 1 - len(context)) * [SOS] + context

    # Trim context to last (n-1)-gram.
    context = tuple(context[-self.n + 1:])

    # Get n-gram.
    ngram = context + (word,)

    # Get counts
    c = self.ngram_count.get(ngram, 0)
    t = self.context_count.get(context, 0)

    # If we have not seen the ngram and λ is 0
    if (c + self.λ == 0):
      return -np.inf

    # Since no OOV word is considered UNK, pretend we have seen all OOV words
    # at least `λ` times, thus increasing alphabet size.
    return log(c + self.λ, 2) - log(t + self.λ * (self.V + numOOV), 2)


class NgramUnk(Ngram):
  def __init__(self, n, λ, voc_ratio):
    self.n = n
    self.λ = λ  # lambda for laplace smoothing.
    self.voc_ratio = voc_ratio  # ratio of train vocabulary that should become UNK
    self.vocabulary = {}
    self.ngram_count = {}  # n-grams
    self.context_count = {}  # (n-1)-grams

  # required, the list of words the language model suports (including EOS)
  def vocab(self):
    return self.vocabulary

  def prefit(self, corpus):
    # Set vocabulary counts.
    self.vocabulary = {}
    for s in corpus:
      for word in s:
        self.vocabulary[word] = self.vocabulary.get(word, 0) + 1
      self.vocabulary[EOS] = self.vocabulary.get(EOS, 0) + 1

    # Sort by most frequent words.
    word_counts = sorted(self.vocabulary.items(), key=lambda wc: -wc[1])

    # Set old vocabulary size.
    self.V_old = len(self.vocabulary)

    # Remove lest frequent words.
    self.vocabulary = set([w for w, _ in word_counts][:int(self.V_old * self.voc_ratio)])

    # Set new vocabulary size.
    self.V = len(self.vocabulary)

    # Replace unk words in corpus by UNK.
    new_corpus = [[w for w in s] for s in corpus]
    for s in range(len(new_corpus)):
      for w in range(len(new_corpus[s])):
        if new_corpus[s][w] not in self.vocabulary:
          new_corpus[s][w] = UNK
    return new_corpus

  # required, update the model when a sentence is observed
  def fit_sentence(self, sentence):
    ngrams = Ngram._get_ngrams(sentence, self.n)
    for ngram in ngrams:
      context = ngram[:-1]
      self.ngram_count[ngram] = self.ngram_count.get(ngram, 0) + 1
      self.context_count[context] = self.context_count.get(context, 0) + 1

  # # optional, if there are any post-training steps (such as normalizing probabilities)
  # def norm(self): pass

  # required, return the log2 of the conditional prob of word, given previous words
  def cond_logprob(self, word, context, numOOV):
    # Add start tokens if not enough context, to model distribution for starting words.
    if len(context) < self.n - 1:
      context = (self.n - 1 - len(context)) * [SOS] + context

    # Trim context to last (n-1)-gram.
    context = tuple(context[-self.n + 1:])

    # Unkify.
    context = tuple(UNK if w not in self.vocabulary else w for w in context)
    if word not in self.vocabulary:
      word = UNK

    # Get n-gram.
    ngram = context + (word,)

    # Get counts
    c = self.ngram_count.get(ngram, 0)
    t = self.context_count.get(context, 0)

    # If we have not seen the ngram and λ is 0
    if (c + self.λ == 0):
      return -np.inf

    # Add 1 to vocabulary size to account for UNK, and use Adjust PPL.
    logprob = log(c + self.λ, 2) - log(t + self.λ * (self.V + 1), 2)

    # Penalize if word is UNK.
    if word == UNK:
      logprob -= log(numOOV, 2)

    return logprob


# class NgramUnkOld(Ngram):
#   def __init__(self, n, λ, unk_ratio):
#     self.n = n
#     self.λ = λ  # lambda for laplace smoothing.
#     self.unk_ratio = unk_ratio  # ratio of train vocabulary that should become UNK
#     self.vocabulary = {}
#     self.ngram_count = {}  # n-grams
#     self.context_count = {}  # (n-1)-grams

#   # required, the list of words the language model suports (including EOS)
#   def vocab(self):
#     return list(self.vocabulary.keys())

#   def prefit(self, corpus):
#     # Set vocabulary counts.
#     self.vocabulary = {}
#     for s in corpus:
#       for word in s:
#         self.vocabulary[word] = self.vocabulary.get(word, 0) + 1
#       self.vocabulary[EOS] = self.vocabulary.get(EOS, 0) + 1

#     # Sort by least frequent words.
#     word_counts = sorted(self.vocabulary.items(), key=lambda wc: wc[1])
#     sum_counts = sum(c for _, c in word_counts)

#     # Get UNKs as the set of least frequent words, which collectively comprise
#     # of, roughly, `unk_ratio` percent.
#     unks = set()
#     unk_count = 0
#     for w, c in word_counts:
#       if w == EOS:
#         continue
#       if unk_count / sum_counts > self.unk_ratio:
#         break
#       unks.add(w)
#       unk_count += c

#     # Set old vocabulary size.
#     self.V_old = len(self.vocabulary)

#     # Remove words which are now UNKs.
#     for word in unks:
#       self.vocabulary.pop(word)

#     # Set vocabulary size.
#     self.V = len(self.vocabulary)

#     # Replace unk words in corpus by UNK.
#     new_corpus = [[w for w in s] for s in corpus]
#     for s in range(len(corpus)):
#       for w in range(len(corpus[s])):
#         if new_corpus[s][w] in unks:
#           new_corpus[s][w] = UNK
#     return new_corpus

#   # required, update the model when a sentence is observed
#   def fit_sentence(self, sentence):
#     ngrams = Ngram._get_ngrams(sentence, self.n)
#     for ngram in ngrams:
#       context = ngram[:-1]
#       self.ngram_count[ngram] = self.ngram_count.get(ngram, 0) + 1
#       self.context_count[context] = self.context_count.get(context, 0) + 1

#   # # optional, if there are any post-training steps (such as normalizing probabilities)
#   # def norm(self): pass

#   # required, return the log2 of the conditional prob of word, given previous words
#   def cond_logprob(self, word, context, numOOV):
#     # Add start tokens if not enough context, to model distribution for starting words.
#     if len(context) < self.n - 1:
#       context = (self.n - 1 - len(context)) * [SOS] + context

#     # Trim context to last (n-1)-gram.
#     context = tuple(context[-self.n + 1:])
#     context = tuple(UNK if w not in self.vocabulary else w for w in context)

#     # Get n-gram.
#     ngram = context + (word,)

#     # Get counts
#     c = self.ngram_count.get(ngram, 0)
#     t = self.context_count.get(context, 0)

#     # If we have not seen the ngram and λ is 0
#     if (c + self.λ == 0):
#       return -np.inf

#     # Add 1 to vocabulary size to account for UNK, and use Adjust PPL.
#     logprob = log(c + self.λ, 2) - log(t + self.λ * (self.V + 1), 2)

#     # Penalize if word is UNK.
#     if word not in self.vocabulary:
#       logprob -= log(numOOV, 2)

#     return logprob

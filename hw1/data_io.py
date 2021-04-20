from sklearn import preprocessing
import tarfile
import os


class Data:
  pass


def load_data(tarfname=''):
  print("-- train data")
  train_data, train_fnames, train_labels = read_tsv("data/speech/train.tsv")
  print(len(train_data))

  print("-- dev data")
  dev_data, dev_fnames, dev_labels = read_tsv("data/speech/dev.tsv")
  print(len(dev_data))

  speech = Data()
  speech.data = train_data + dev_data
  speech.fnames = train_fnames + dev_fnames
  speech.labels = train_labels + dev_labels

  # Labels
  speech.le = preprocessing.LabelEncoder()
  speech.le.fit(speech.labels)
  speech.target_labels = speech.le.classes_
  speech.y = speech.le.transform(speech.labels)
  return speech


def load_unlabeled_data():
  """Reads the unlabeled data.

  The returned object contains three fields that represent the unlabeled data.

  data: documents, represented as sequence of words
  fnames: list of filenames, one for each document
  X: bag of word vector for each document, using the speech.vectorizer
  """
  print("Reading unlabeled data")
  unlabeled = Data()
  unlabeled.data = []
  unlabeled.fnames = []
  for m in sorted(os.listdir('data/speech/unlabeled')):
    unlabeled.fnames.append(m)
    unlabeled.data.append(read_instance(f'unlabeled/{m}'))
  return unlabeled



def read_tsv(filename):
  with open(filename) as tsvfile:
    data = []
    labels = []
    fnames = []
    for line in tsvfile.readlines():
      (ifname, label) = line.strip().split("\t")
      content = read_instance(ifname)
      labels.append(label)
      fnames.append(ifname)
      data.append(content)
    return data, fnames, labels


def read_instance(ifname):
  with open(f'data/speech/{ifname}') as f:
    return f.read().strip()
  raise "Could not open file"



def file_to_id(fname):
  return str(int(fname.replace("unlabeled/", "").replace("labeled/", "").replace(".txt", "")))


def write_pred_kaggle_file(y, outfname, le):
  """Writes the predictions in Kaggle format.

  Given the unlabeled object, classifier, outputfilename, and the speech object,
  this function write the predictions of the classifier on the unlabeled data and
  writes it to the outputfilename. The speech object is required to ensure
  consistent label names.
  """
  # print("Making predictions")
  # yp = cls.predict(unlabeled.X)
  labels = le.inverse_transform(y)
  print(f"Writing to {outfname}")
  f = open(outfname, 'w')
  f.write("FileIndex,Category\n")
  for i in range(len(labels)):
    # for i in range(len(unlabeled.fnames)):
    # fname = unlabeled.fnames[i]
    # iid = file_to_id(fname)
    f.write(str(i + 1))
    f.write(",")
    # f.write(fname)
    # f.write(",")
    f.write(labels[i])
    f.write("\n")
  f.close()


def write_basic_kaggle_file(tsvfile, outfname):
  """Writes the output Kaggle file of the naive baseline.

  This baseline predicts OBAMA_PRIMARY2008 for all the instances.
  You will not be able to run this code, since the tsvfile is not
  accessible to you (it is the test labels).
  """
  print(f"Writing pred file into {outfname}")
  f = open(outfname, 'w')
  f.write("FileIndex,Category\n")
  i = 0
  with open(tsvfile, 'r') as tf:
    for line in tf:
      (ifname, label) = line.strip().split("\t")
      i += 1
      f.write(str(i))
      f.write(",")
      f.write("OBAMA_PRIMARY2008")
      f.write("\n")
  f.close()


def config_repr(config):
  return ','.join(config['type']) + '__' + '__'.join([f"{k}={','.join([str(x) for x in v])}" for k, v in config.items() if k != 'type'])

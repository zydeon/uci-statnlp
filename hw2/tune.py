from data import read_texts
from lm import Unigram, NgramNoUnk, NgramUnk
import numpy as np
import pandas as pd
import json
import sys
import os
import time


def tune_nounk(outfile, n, λs):
  dnames = ["brown", "reuters", "gutenberg"]
  data = {}
  for dname in dnames:
    print("-----------------------")
    print(dname)
    data[dname] = read_texts("data/corpora.tar.gz", dname)

  results = []
  for λ in λs:
    ppls = []
    for dname in dnames:
      model = NgramNoUnk(n, λ)
      print(f"Training on {dname}\tλ = {λ}...", end='')
      model.fit_corpus(data[dname].train)
      print('done')

      print('  computing perplexity...', end='')
      ppls.append(model.perplexity(data[dname].dev))
      print(f'done. PPL={ppls[-1]}')

    avg_in_domain_ppl = np.mean(ppls)
    print(f'Avg in-domain: {avg_in_domain_ppl}\n')
    results.append((n, λ) + tuple(ppls) + (avg_in_domain_ppl,))

    # Save what we have already
    df = pd.DataFrame(results, columns=('n', 'λ') + tuple(dnames) + ('avg',))
    df.to_csv(outfile, index=False)


# def tune_unk(outfile, n, λs, unk_ratios):
#   dnames = ["brown", "reuters", "gutenberg"]
#   data = {}
#   for dname in dnames:
#     print("-----------------------")
#     print(dname)
#     data[dname] = read_texts("data/corpora.tar.gz", dname)

#   results = []
#   for λ in λs:
#     for unk_ratio in unk_ratios:
#       ppls = []
#       for dname in dnames:
#         model = NgramUnk(n, λ, unk_ratio)
#         print(f"Training on {dname}\tλ = {λ:.7f}\tunk_ratio={unk_ratio:.7f}...", end='')
#         model.fit_corpus(data[dname].train)
#         print(f'done\tV={model.V}')
#         print('  computing perplexity...', end='')
#         ppls.append(model.perplexity(data[dname].dev))
#         print(f'done. PPL={ppls[-1]}')

#       avg_in_domain_ppl = np.mean(ppls)
#       print(f'Avg in-domain: {avg_in_domain_ppl}\n')
#       results.append((n, λ, unk_ratio) + tuple(ppls) + (avg_in_domain_ppl,))

#       # Save what we have already
#       df = pd.DataFrame(results, columns=('n', 'λ', 'unk_ratio') + tuple(dnames) + ('avg',))
#       df.to_csv(outfile, index=False)


def tune_unk_voc_ratios(outfile, n, λs, voc_ratios):
  dnames = ["brown", "reuters", "gutenberg"]
  data = {}
  for dname in dnames:
    print("-----------------------")
    print(dname)
    data[dname] = read_texts("data/corpora.tar.gz", dname)

  results = []
  for λ in λs:
    for voc_ratio in voc_ratios:
      ppls = []
      for dname in dnames:
        model = NgramUnk(n, λ, voc_ratio)
        print(f"Training on {dname}\tλ = {λ:.7f}\tvoc_ratio={voc_ratio:.7f}...", end='')
        model.fit_corpus(data[dname].train)
        print(f'done\tV={model.V}\t({model.V_old})')
        print('  computing perplexity...', end='')
        ppls.append(model.perplexity(data[dname].dev))
        print(f'done. PPL={ppls[-1]}')

      avg_in_domain_ppl = np.mean(ppls)
      print(f'Avg in-domain: {avg_in_domain_ppl}\n')
      results.append((n, λ, voc_ratio) + tuple(ppls) + (avg_in_domain_ppl,))

      # Save what we have already
      df = pd.DataFrame(results, columns=('n', 'λ', 'voc_ratio') + tuple(dnames) + ('avg',))
      df.to_csv(outfile, index=False)


def check_params(params):
  if len(params) != 2 or (params[1] not in ('unk', 'nounk')):
    print(f'Usage: {params[0]} <unk|no_unk> < config.json')
    sys.exit(-1)


if __name__ == '__main__':
  params = sys.argv
  check_params(params)
  config = json.load(sys.stdin)

  # Create output folder.
  print("Creating results folder")
  ts = int(time.time())
  model_type = params[1]
  model_type_str = f'nounk'
  if model_type == 'unk_old':
    model_type_str = f"unk={','.join(map(str, eval(config['unk_ratios'])))}"
  if model_type == 'unk':
    model_type_str = f"voc={','.join(map(str, eval(config['voc_ratios'])))}"
  out_folder = f"results/tuner/n={config['n']}__{model_type_str}__{ts}"
  os.makedirs(out_folder)

  # Get results.
  if model_type == 'unk':
    results = tune_unk_voc_ratios(f'{out_folder}/results.csv',
                                  config['n'],
                                  eval(config['λs']),
                                  eval(config['voc_ratios']))
  else:
    results = tune_nounk(f'{out_folder}/results.csv', config['n'], eval(config['λs']))

  # Save results.
  # print(f"Saving results to {out_folder}/results.csv")
  # results.to_csv(f'{out_folder}/results.csv', index=False)

  # params = sys.argv
  # check_params(params)

  # model = params[1]
  # if model == 'unigram':
  #   run_model(lambda: Unigram(), 'results/unigram')
  # else:
  #   n = int(params[2])
  #   λ = float(params[3])
  #   if len(params) == 5:
  #     unk_ratio = float(params[4])
  #     run_model(lambda: NgramUnk(n, λ, unk_ratio), f'results/ngram_n={n}_l={λ}_unk={unk_ratio}')
  #   else:
  #     run_model(lambda: NgramNoUnk(n, λ), f'results/ngram_n={n}_l={λ}')

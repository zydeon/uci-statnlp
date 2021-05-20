from data import run_model
from lm import Unigram, NgramNoUnk, NgramUnk
import sys


def check_params(params):
  if (
      (len(params) <= 1)
      or (params[1] not in ('unigram', 'ngram'))
      or (params[1] == 'ngram' and len(params) < 4)
  ):
    print(f'Usage: {params[0]} <unigram|ngram> [n λ [voc_ratio]]')
    sys.exit(-1)


if __name__ == '__main__':
  params = sys.argv
  check_params(params)

  model = params[1]
  if model == 'unigram':
    run_model(lambda: Unigram(), 'results/unigram')
  else:
    n = int(params[2])
    λ = float(params[3])
    if len(params) == 5:
      voc_ratio = float(params[4])
      run_model(lambda: NgramUnk(n, λ, voc_ratio), f'results/ngram_n={n}_l={λ}_voc={voc_ratio}')
    else:
      run_model(lambda: NgramNoUnk(n, λ), f'results/ngram_n={n}_l={λ}')

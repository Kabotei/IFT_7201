import numpy
from matplotlib import pyplot
from os import path
from pathlib import Path
from tqdm import tqdm
from bandit import NormalBandit
from utils import means_100
from q3.a import TSNormal
from .c1 import BayesUCBNormalBatch, TSNormalBatch
from .a import BayesUCBNormal


T = 300
sigma = 0.1
buffer_sizes = [10, 50, 100]

algorithms = {
  'TS': [
    lambda _: TSNormal(sigma=sigma),
    lambda buffer_size: TSNormalBatch(sigma=sigma, batch_size=buffer_size),
  ],
  'Bayes UCB': [
    lambda _: BayesUCBNormal(sigma=sigma, c=5),
    lambda buffer_size: BayesUCBNormalBatch(sigma=sigma, c=5, batch_size=buffer_size),
  ],
}

fig_tqdm = tqdm(total=len(algorithms))
buffer_tqdm = tqdm(total=len(buffer_sizes))

for fig_nb, (alg_type, algos) in enumerate(algorithms.items()):
  buffer_tqdm.reset()
  pyplot.figure(figsize=(7,9))
  pyplot.subplots_adjust(hspace=0.3)
  pyplot.suptitle(f'Différence entre {alg_type} sans buffer et avec buffer')

  for buffer_size in buffer_sizes:
    all_regrets = []

    for algo_func in algos:
      regrets = []

      for i, means in enumerate(means_100):
        bandit = NormalBandit(means=means, sigma=sigma, seed=i)
        algo_func(buffer_size).execute(bandit=bandit, T=T, seed=i)
        regrets.append(bandit.get_cumulative_regret())

      all_regrets.append(numpy.array(regrets))

    diff = all_regrets[1] - all_regrets[0]

    pyplot.subplot(2, 1, 1)
    pyplot.title('Différences')
    pyplot.xlabel('Pas de temps')
    pyplot.ylabel('Différence de pseudo-regret cumulatif')
    diff_mean = numpy.mean(diff, axis=0)
    diff_std = numpy.std(diff, axis=0)
    pyplot.plot(diff_mean, label=f'L={buffer_size}')
    pyplot.fill_between(numpy.arange(T), diff_mean, diff_mean + diff_std, alpha=0.4)
    pyplot.legend()

    pyplot.subplot(2, 1, 2)
    pyplot.title('Différences en %')
    pyplot.xlabel('Pas de temps')
    pyplot.ylabel('Différence de pseudo-regret cumulatif (%)')
    mean_0 = numpy.mean(all_regrets[0], axis=0)
    diff_mean_pct = diff_mean / mean_0 * 100
    diff_std_pct = diff_std / mean_0 * 100
    pyplot.plot(diff_mean_pct, label=f'L={buffer_size}')
    pyplot.fill_between(numpy.arange(T), diff_mean_pct, diff_mean_pct + diff_std_pct, alpha=0.4)
    pyplot.legend()

    buffer_tqdm.update()

  pyplot.savefig(path.join(Path.cwd(), 'q4', f'c-{fig_nb}.png'))
  fig_tqdm.update()


buffer_tqdm.display()
fig_tqdm.display()

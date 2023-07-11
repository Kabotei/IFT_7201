import numpy
from tqdm import tqdm
from matplotlib import pyplot
from os import path
from pathlib import Path
from bandit import NormalBandit
from utils import means_100
from .a import TSAgrawal2, TSNormal


T = 1000
sigmas = [0.1, 0.5, 1]

algorithms = [
  TSAgrawal2(),
  *[TSNormal(sigma=sigma) for sigma in sigmas]
]

results = []

for algo in tqdm(algorithms):
  regrets = []

  for i, means in enumerate(means_100):
    bandit = NormalBandit(means=means, sigma=1, seed=i)
    algo.execute(bandit=bandit, T=T, seed=i)
    regrets.append(bandit.get_cumulative_regret())

  avg_regret = numpy.mean(regrets, axis=0)
  std_regret = numpy.std(regrets, axis=0)

  results.append({
    'name': algo.get_name(),
    'mean': avg_regret,
    'std': std_regret
  })

for fig_nb, result in enumerate(results[1:]):
  pyplot.figure()
  pyplot.title('Comparaison du pseudo-regret cumulatif de\nplusieurs algorithmes TS')
  pyplot.xlabel('Pseudo-regret cumulatif')
  pyplot.ylabel('Pas de temps')

  pyplot.plot(results[0]['mean'], label=results[0]['name'])
  pyplot.fill_between(numpy.arange(T), results[0]['mean'], results[0]['mean'] + results[0]['std'], alpha=0.4)

  pyplot.plot(result['mean'], label=result['name'])
  pyplot.fill_between(numpy.arange(T), result['mean'], result['mean'] + result['std'], alpha=0.4)

  pyplot.legend()
  pyplot.savefig(path.join(Path.cwd(), 'q3', f'b-{fig_nb}.png'))

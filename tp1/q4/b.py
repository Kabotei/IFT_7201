import numpy
from matplotlib import pyplot
from tqdm import tqdm
from os import path
from pathlib import Path
from bandit import NormalBandit
from utils import means_100
from .a import BayesUCBNormal
from q3.a import TSNormal


T = 1000
sigma = 0.5

algorithms = [
  BayesUCBNormal(sigma=sigma, c=5),
  TSNormal(sigma=sigma)
]

for algo in tqdm(algorithms):
  regrets = []

  for i, means in enumerate(means_100):
    bandit = NormalBandit(means=means, sigma=sigma, seed=i)
    algo.execute(bandit=bandit, T=T, seed=i)
    regrets.append(bandit.get_cumulative_regret())

  avg_regret = numpy.mean(regrets, axis=0)
  std_regret = numpy.std(regrets, axis=0)

  pyplot.plot(avg_regret, label=algo.get_name())
  pyplot.fill_between(numpy.arange(T), avg_regret, avg_regret + std_regret, alpha=0.4)

pyplot.title('Différence de convergence entre Bayes-UCB normal et TS normal')
pyplot.xlabel('Pas de temps')
pyplot.ylabel('Pseudo-regret cumulatif')
pyplot.legend()
pyplot.savefig(path.join(Path.cwd(), 'q4', 'b.png'))

import numpy
from matplotlib import pyplot
from os import path
from pathlib import Path
from bandit import BernoulliBandit
from .a import KLUCBNormal, KLUCBBernoulli


N = 10
T = 5000
c = 4
sigma = 0.5

configurations = numpy.array([
  [0.1, 0.9],
  [0.4, 0.6],
  [0.45, 0.55]
])

algorithms = [
  KLUCBNormal(sigma=sigma, c=c),
  KLUCBBernoulli(sigma=sigma, c=c)
]

for fig_nb, means in enumerate(configurations):
  pyplot.figure()
  pyplot.title('Comparaison du pseudo-regret cumulatif avec $\mu$ = ' + str(means))
  pyplot.xlabel('Pas de temps (t)')
  pyplot.ylabel('Regret pseudo-cumulatif')

  for algo in algorithms:
    regrets = []

    for i in range(N):
      bandit = BernoulliBandit(means=means, seed=i)
      algo.execute(bandit=bandit, T=T)
      regrets.append(bandit.get_cumulative_regret())

    avg_regret = numpy.mean(regrets, axis=0)
    std_regret = numpy.std(regrets, axis=0)

    pyplot.plot(avg_regret, label=algo.get_name())
    pyplot.fill_between(numpy.arange(T), avg_regret, avg_regret + std_regret, alpha=0.4)

  pyplot.legend()
  pyplot.savefig(path.join(Path.cwd(), 'q2', f'b-{fig_nb}.png'))

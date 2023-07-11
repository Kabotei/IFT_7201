import numpy
from tqdm import tqdm
from os import path
from pathlib import Path
from matplotlib import pyplot
from bandit import NormalBandit
from .b import ETC, ETCOptimal, UCB1


N = 50
T = 1000
n_gaps = 50
sigma = 1

gaps = numpy.linspace(numpy.sqrt(4 * sigma**2 / T) + 0.00001, 1.0, n_gaps, dtype=float)

algorithms = [
  lambda bandit, T: ETCOptimal(bandit=bandit, T=T),
  lambda *_: ETC(m=25),
  lambda *_: UCB1(),
]

alg_bar = tqdm(total=len(algorithms))
instance_bar = tqdm(total=gaps.size)

for algo_factory in algorithms:
  instance_bar.reset()
  all_regrets = []

  for i in range(N):
    instance_regrets = []

    for gap in gaps:
      means = numpy.array([0.5 - gap/2, 0.5 + gap/2])

      bandit = NormalBandit(means=means, sigma=sigma, seed=i)
      algo = algo_factory(bandit, T)

      algo.execute(bandit, T)
      regret = bandit.get_cumulative_regret()[-1]

      instance_regrets.append(regret)
    
    all_regrets.append(instance_regrets)
    instance_bar.update()

  mean = numpy.mean(all_regrets, axis=0)
  # std = numpy.std(all_regrets, axis=0)

  pyplot.plot(gaps, mean, label=algo.get_name())
  # pyplot.fill_between(gaps, mean - std, mean + std, alpha=0.4)

  alg_bar.update()

alg_bar.refresh()
instance_bar.refresh()

pyplot.title('Regret espéré en fonction des gaps de\nsous-optimalité, selon plusieurs algorithmes')
pyplot.xlabel('$\Delta$')
pyplot.ylabel('Expected regret')
pyplot.legend()
pyplot.savefig(path.join(Path.cwd(), 'q1', 'c.png'))

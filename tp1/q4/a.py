import numpy
from scipy import stats
from abstract import Bandit, BanditAlgorithm


class BayesUCBNormal(BanditAlgorithm):
  def __init__(self, sigma: float, c: int):
    self.__sigma = sigma
    self.__c = c

  def execute(self, bandit: Bandit, T: int, *_, **__):
    K = bandit.get_K()
    sums, ssums, plays = numpy.zeros(K), numpy.zeros(K), numpy.zeros(K)
    log_T_pow_c = numpy.power(numpy.log(T), self.__c)

    for t in range(T):
      if t < K:
        k_t = t
      else:
        q = 1 - 1 / ((t+1) * log_T_pow_c)
        prior = 1 / self.__sigma * (plays - 1)
        Q = stats.t.ppf(q, prior)
        ucbs = sums / plays + numpy.sqrt(ssums / plays) * Q
        k_t = numpy.argmax(ucbs)

      reward = bandit.play(k_t)

      sums[k_t] += reward
      ssums[k_t] += reward ** 2
      plays[k_t] += 1

  def get_name(self) -> str:
    return f'Bayes-UCB Normal (sigma={self.__sigma})'

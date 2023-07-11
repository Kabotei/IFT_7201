import numpy
from abc import ABC, abstractmethod
from abstract import Bandit, BanditAlgorithm
from .kl import klucb_upper_bisection


class KLUCB(BanditAlgorithm, ABC):
  def __init__(self, sigma: float, c: int):
    self.sigma = sigma
    self.c = c

  def execute(self, bandit: Bandit, T: int):
    K = bandit.get_K()
    sums, plays = numpy.zeros(K), numpy.zeros(K)

    for t in range(T):
      if t < K:
        k_t = t
      else:
        ucbs = self.get_ucbs(t=t, K=K, plays=plays, sums=sums)
        k_t = numpy.argmax(ucbs)

      reward = bandit.play(k_t)

      sums[k_t] += reward
      plays[k_t] += 1

  @abstractmethod
  def get_ucbs(self, t: int, K: int, plays: numpy.ndarray, sums: numpy.ndarray) -> float:
    raise NotImplementedError()

  @abstractmethod
  def get_name(self) -> str:
    raise NotImplementedError()


class KLUCBNormal(KLUCB):
  def get_ucbs(self, t: int, K: int, plays: numpy.ndarray, sums: numpy.ndarray) -> float:
    log_t = numpy.log(t + 1)

    return sums / plays + numpy.sqrt(
      2 * self.sigma * (log_t + self.c * numpy.log(log_t)) / plays
    )

  def get_name(self) -> str:
    return 'kl-UCB (normal)'


class KLUCBBernoulli(KLUCB):
  def get_ucbs(self, t: int, K: int, plays: numpy.ndarray, sums: numpy.ndarray) -> float:
    ucbs = numpy.zeros(K)

    for k in range(K):
      ucbs[k] = klucb_upper_bisection(N=plays, S=sums, k=k, t=t)

    return ucbs

  def get_name(self) -> str:
    return 'kl-UCB (bernoulli)'

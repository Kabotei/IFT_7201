import numpy
from abc import ABC, abstractmethod
from abstract import Bandit, BanditAlgorithm


class ThompsonSampling(BanditAlgorithm, ABC):
  def execute(self, bandit: Bandit, T: int, seed: int = None):
    K = bandit.get_K()
    sums, plays = numpy.zeros(K), numpy.zeros(K)
    random = numpy.random.RandomState(seed=seed)

    for t in range(T):
      samples = self.get_samples(random, sums, plays)
      k_t = numpy.argmax(samples)

      reward = bandit.play(k_t)

      sums[k_t] += reward
      plays[k_t] += 1

  @abstractmethod
  def get_samples(self, random: numpy.random.RandomState, sums: numpy.ndarray, plays: numpy.ndarray) -> numpy.ndarray:
    raise NotImplementedError()


class TSBeta(ThompsonSampling):
  def __init__(self, alpha: float, beta: float):
    super().__init__()
    self.__alpha = alpha
    self.__beta = beta

  def get_samples(self, random: numpy.random.RandomState, sums: numpy.ndarray, plays: numpy.ndarray) -> numpy.ndarray:
    return random.beta(self.__alpha + sums, self.__beta + plays - sums)

  def get_name(self) -> str:
    return 'TS with Beta prior'


class TSAgrawal2(ThompsonSampling):
  def get_samples(self, random: numpy.random.RandomState, sums: numpy.ndarray, plays: numpy.ndarray) -> numpy.ndarray:
    plays += 1
    return random.normal(sums/plays, 1/plays)

  def get_name(self) -> str:
    return 'TS Agrawal [2]'


class TSNormal(ThompsonSampling):
  def __init__(self, sigma: float):
    super().__init__()
    self.__sigma = sigma

  def get_samples(self, random: numpy.random.RandomState, sums: numpy.ndarray, plays: numpy.ndarray) -> numpy.ndarray:
    plays += 1
    return random.normal(sums/plays, self.__sigma) # TODO verify

  def get_name(self) -> str:
    return f'TS normal (sigma={self.__sigma})'

import numpy
from abstract import Bandit, BanditAlgorithm
from scipy import stats


class BayesUCBNormalBatch(BanditAlgorithm):
  def __init__(self, sigma: float, c: int, batch_size: int):
    if batch_size < 1:
      raise ValueError('batch size must be at least 1')

    self.__sigma = sigma
    self.__c = c
    self.__batch_size = batch_size

  def execute(self, bandit: Bandit, T: int, *_, **__):
    K = bandit.get_K()
    sums, ssums, plays = numpy.zeros(K), numpy.zeros(K), numpy.zeros(K)
    log_T_pow_c = numpy.power(numpy.log(T), self.__c)

    rewards = []

    for t in range(T):
      update_rewards = len(rewards) == self.__batch_size

      if t < K:
        k_t = t
        update_rewards = True
      else:
        q = 1 - 1 / ((t+1) * log_T_pow_c)
        prior = 1 / self.__sigma * (plays - 1)
        Q = stats.t.ppf(q, prior)
        ucbs = sums / plays + numpy.sqrt(ssums / plays) * Q
        k_t = numpy.argmax(ucbs)

      rewards.append(bandit.play(k_t))

      if update_rewards:
        for reward in rewards:
          sums[k_t] += reward
          ssums[k_t] += reward ** 2
          plays[k_t] += 1

        rewards = []

  def get_name(self) -> str:
    return f'Bayes-UCB Normal Batch (sigma={self.__sigma})'


class TSNormalBatch(BanditAlgorithm):
  def __init__(self, sigma: float, batch_size: int):
    if batch_size < 1:
      raise ValueError('batch size must be at least 1')

    self.__sigma = sigma
    self.__batch_size = batch_size

  def execute(self, bandit: Bandit, T: int, seed: int = None):
    K = bandit.get_K()
    sums, plays = numpy.zeros(K), numpy.zeros(K)
    random = numpy.random.RandomState(seed=seed)

    rewards = []

    for _ in range(T):
      samples = self.get_samples(random, sums, plays)
      k_t = numpy.argmax(samples)

      rewards.append(bandit.play(k_t))

      if len(rewards) == self.__batch_size:
        for reward in rewards:
          sums[k_t] += reward
          plays[k_t] += 1

        rewards = []

  def get_samples(self, random: numpy.random.RandomState, sums: numpy.ndarray, plays: numpy.ndarray) -> numpy.ndarray:
    plays += 1
    return random.normal(sums/plays, self.__sigma)

  def get_name(self) -> str:
    return f'TS normal Batch (sigma={self.__sigma})'

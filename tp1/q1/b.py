import numpy
from abstract import BanditAlgorithm, Bandit


class ETC(BanditAlgorithm):
  def __init__(self, m: int = None):
    self.m = m

  def execute(self, bandit: Bandit, T: int):
      K = bandit.get_K()
      rewards = numpy.zeros(K)

      for k in range(K):
        for i in range(self.m):
          rewards[k] += bandit.play(k)

      k_star = numpy.argmax(rewards)

      for t in range(T - K * self.m):
        bandit.play(k_star)

      return k_star

  def get_name(self) -> str:
    return f'etc (m={self.m})'


class ETCOptimal(ETC):
  def __init__(self, bandit: Bandit, T: int):
    super().__init__(m=0)

    sigma_2 = numpy.square(bandit.sigma)
    gap_2 = numpy.square(numpy.sort(bandit.gaps)[1])
    log_expr = T * gap_2 / (4 * sigma_2)

    if log_expr <= 1:
      raise ValueError(f'T * gap^2 (={T*gap_2}) must be greater than 4*sigma^2 (={4*sigma_2})')

    optimal_m = int(4 * sigma_2 / gap_2 * numpy.log(log_expr))

    self.m = optimal_m

  def get_name(self) -> str:
    return 'etc (m optimal)'


class UCB1(BanditAlgorithm):
  def execute(self, bandit: Bandit, T: int):
    '''Play the given bandit over T rounds using the UCB1 strategy.'''
    K = bandit.get_K()
    sums, plays = numpy.zeros(K), numpy.zeros(K)

    for t in range(T):
      if t < K:
        k_t = t
      else:
        log_t = numpy.log(t + 1)
        ucbs = sums / plays + numpy.sqrt(2 * log_t / plays)
        k_t = numpy.argmax(ucbs)

      reward = bandit.play(k_t)

      sums[k_t] += reward
      plays[k_t] += 1

  def get_name(self) -> str:
    return 'UCB1'

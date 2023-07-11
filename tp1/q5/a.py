import numpy
from abc import ABC, abstractmethod
from typing import List, Any
from abstract import Bandit


class ContextualBandit(Bandit):
  @abstractmethod
  def get_context(self) -> float:
    raise NotImplementedError()


class ContextualBanditAlgorithm(ABC):
  @abstractmethod
  def execute(self, bandit: ContextualBandit, T: int):
    raise NotImplementedError()

  @abstractmethod
  def get_name(self) -> str:
    raise NotImplementedError()


class ContextualLinGaussianBandit(ContextualBandit):
  def __init__(self, thetas, noise, seed=None):
    self.thetas = numpy.copy(thetas)
    self.sigma = noise
    self.random = numpy.random.RandomState(seed)
    
    self.K = thetas.shape[0]
    self.d = thetas.shape[1]
    
    self.context = None
    
    self.regret = []

  def get_K(self) -> int:
    return self.K

  def get_context(self) -> float:
    self.context = self.random.uniform(-1, 1)
    return self.context

  def play(self, k) -> float:
    phi_s = numpy.array([1, self.context, self.context**2, self.context**3, self.context**4])
    means = self.thetas.dot(phi_s)
    k_star = numpy.argmax(means)
    self.regret.append(means[k_star] - means[k])
    return means[k] + self.random.normal(0, self.sigma)

  def get_cumulative_regret(self) -> numpy.ndarray:
    return numpy.cumsum(self.regret)


class KernelTS(ContextualBanditAlgorithm):
  def __init__(self, Lambda: float, R: float, S: float, models: List[Any], seed: int = None):
    '''
    Lambda: regularization
    R: upper bound sur sigma
    S: upper bound sur norm_2(theta)
    models: K Gaussian Process models (one per action)
    '''
    self.__Lambda = Lambda
    self.__R = R
    self.__S = S
    self.__models = models
    self.__random = numpy.random.RandomState(seed=seed)

  def execute(self, bandit: ContextualBandit, T: int):
    K = bandit.get_K()
    S_t, y_t = [[] for k in range(K)], [[] for k in range(K)]

    for t in range(T):
      s_t = bandit.get_context()

      samples = []

      for model in self.__models:
        f_hat, sqrt_k = model.predict([[s_t]], return_std=True)
        Sigma = self.__R**2 / self.__Lambda * sqrt_k
        samples.append(self.__random.normal(f_hat, Sigma))

      k_t = numpy.argmax(samples)

      r_t = bandit.play(k_t)
      S_t[k_t].append([s_t])
      y_t[k_t].append(r_t)

      self.__models[k_t].fit(numpy.array(S_t[k_t]), numpy.array(y_t[k_t]))

  def get_name(self) -> str:
    return f'Kernel TS (R={self.__R})'

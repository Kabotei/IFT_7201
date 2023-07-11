from abc import ABC, abstractmethod
import numpy


class Bandit(ABC):
  @abstractmethod
  def get_K(self) -> int:
    raise NotImplementedError()

  @abstractmethod
  def play(self, k: int, dry_run: bool = False) -> float:
    raise NotImplementedError()

  @abstractmethod
  def get_cumulative_regret(self) -> numpy.ndarray:
    raise NotImplementedError()


class BanditAlgorithm(ABC):
  @abstractmethod
  def execute(self, bandit: Bandit, T: int):
    raise NotImplementedError()

  @abstractmethod
  def get_name(self) -> str:
    raise NotImplementedError()

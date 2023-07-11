import numpy as np
from typing import List


class UpdatableValue:
    def __init__(self, initial_value: float, decay: float, min_value: float = 0) -> None:
        self.__initial_value = initial_value
        self.__decay = decay
        self.__min_value = min_value

        self.__value = initial_value

    def step(self) -> None:
        self.__value = max(self.__min_value, self.__value * (1 - self.__decay))

    def unstep(self, n: int = 1):
        for _ in range(n):
            self.__value /= (1 - self.__decay)

    def get(self) -> float:
        return self.__value
        
    def reset(self, value: float = None):
        self.__value = value if value else self.__initial_value


class MultiUpdatableValue:
    def __init__(self, updatable_values: List[UpdatableValue], probs: List[float] = None, seed: int = 42) -> None:
        if probs == None:
            probs = [1/len(updatable_values)] * len(updatable_values)
        elif len(probs) != len(updatable_values):
            raise ValueError('mismatching lists lengths')

        self.__updatable_values = updatable_values
        self.__probs = probs
        self.__indices = list(range(len(probs)))
        self.__random = np.random.RandomState(seed=seed)

    def get(self) -> UpdatableValue:
        index = self.__random.choice(self.__indices, p=self.__probs)
        return self.__updatable_values[index]


class StringBuffer:
    def __init__(self) -> None:
        self.__buffer = ''

    def add(self, text: str):
        self.__buffer += f'{text}\n'

    def print(self):
        print(self.__buffer)
        self.__buffer = ''

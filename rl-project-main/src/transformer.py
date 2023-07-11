from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.env import Action, State
from src.graph import Node


class Normalizer:
    def __init__(self, mins: List[float], maxs: List[float]):
        if len(mins) != len(maxs):
            raise ValueError("mins and maxs should have same dimension.")

        mins = np.array(mins)
        maxs = np.array(maxs)

        self.__means = (maxs + mins) / 2
        self.__stds = (maxs - mins) / 4

    def normalize(self, value: np.ndarray):
        return (value - self.__means) / self.__stds

    def denormalize(self, normalized_value: np.ndarray):
        return normalized_value * self.__stds + self.__means


# TODO not sure it would work... embedding to 1D loses lots of info
class Vecor2DTo1DEmbedder:
    def __init__(self, mins: List[float], maxs: List[float]):
        if len(mins) != len(maxs):
            raise ValueError("mins and maxs should have same dimension.")

        self.__mins = np.array(mins)
        self.__maxs = np.array(maxs)

    def embed(self, input: np.ndarray) -> np.ndarray:
        pass
        # TODO
        # if len(input) != 2:
        #     raise ValueError("input dimension should be 2.")

        # return np.array([input[0] + input[1] - ])


class InputTransformer(ABC):
    @abstractmethod
    def to_net_input(self, state: State, action: Action) -> np.ndarray:
        raise NotImplementedError()

    def input_size(self) -> int:
        raise NotImplementedError()


class BaseInput(InputTransformer):
    def __init__(self, normalizer: Normalizer):
        self.__normalizer = normalizer

    def to_net_input(self, state: State, action: Action) -> np.ndarray:
        current_node = self.__normalizer.normalize(state.current_node.to_numpy())
        end_node = self.__normalizer.normalize(state.end_node.to_numpy())
        action = self.__normalizer.normalize(action.to_numpy())

        return np.array([*current_node, *end_node, *action])

    def input_size(self) -> int:
        return 6


class InputWithActionWeight(InputTransformer):
    def __init__(self, normalizer: Normalizer):
        self.__normalizer = normalizer

    def to_net_input(self, state: State, action: Action) -> np.ndarray:
        current_node = self.__normalizer.normalize(state.current_node.to_numpy())
        end_node = self.__normalizer.normalize(state.end_node.to_numpy())
        normalized_action = self.__normalizer.normalize(action.to_numpy())

        return np.array([*current_node, *end_node, *normalized_action, action.weight])

    def input_size(self) -> int:
        return 7


class InputWithPreviousNode(InputTransformer):
    def __init__(self, normalizer: Normalizer):
        self.__normalizer = normalizer

    def to_net_input(self, state: State, action: Action) -> np.ndarray:
        previous_node = self.__normalizer.normalize(state.previous_node.to_numpy())
        current_node = self.__normalizer.normalize(state.current_node.to_numpy())
        end_node = self.__normalizer.normalize(state.end_node.to_numpy())
        action = self.__normalizer.normalize(action.to_numpy())

        return np.array([*previous_node, *current_node, *end_node, *action])

    def input_size(self) -> int:
        return 8


# TODO not sure it would work... embedding to 1D loses lots of info
class CompactTransformer:
    def __init__(self, normalizer: Normalizer, embedder: Vecor2DTo1DEmbedder):
        self.__normalizer = normalizer
        self.__embedder = embedder

    def to_net_input(self, state: State, action: Node) -> np.ndarray:
        current_node = self.__embedder.embed(self.__normalizer.normalize(state.current_node.to_numpy()))
        end_node = self.__embedder.embed(self.__normalizer.normalize(state.end_node.to_numpy()))
        action = self.__embedder.embed(self.__normalizer.normalize(action.to_numpy()))

        return np.array([*current_node, *end_node, *action])

    def input_size(self) -> int:
        return 3

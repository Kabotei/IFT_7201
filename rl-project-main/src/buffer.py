import random
from typing import List, NamedTuple
from src.env import Action, State


class Sequence(NamedTuple):
    state: State
    choosen_action: Action
    reward: float
    next_state: State
    trajectory_done: bool


class ReplayBuffer:
    def __init__(self, buffer_size: int, data: List[Sequence] = []):
        self.__list = data
        self.__max_size = buffer_size

    def store(self, element: Sequence):
        '''
        Stores an element. If the replay buffer is already full, deletes the oldest
        element to make space.
        '''
        self.__list.append(element)

        if len(self.__list) > self.__max_size:
          del self.__list[0]

    def get_batch(self, batch_size: int) -> List[Sequence]:
        '''
        Returns a list of batch_size elements from the buffer.
        '''
        return random.sample(self.__list, batch_size)

    def __len__(self) -> int:
        return len(self.__list)

    def get_list(self) -> List[Sequence]:
        return self.__list

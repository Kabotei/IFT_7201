import multiprocessing
from multiprocessing.context import Process
from queue import Empty
from typing import Generic, Iterator, List, TypeVar
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ExperimentResult:
    _lambda: float
    rewards: np.ndarray


@dataclass
class ExperimentResults:
    max_trajectories: int
    runs: int
    results: List[ExperimentResult] = field(default_factory=list)

class ProcessExecutor:
    def __init__(self, processes: List[Process]):
        self.__processes = processes

    def execute(self, block_until_finished: bool = False):
        for process in self.__processes:
            process.start()

        if block_until_finished:
            for process in self.__processes:
                process.join()
                print('A process has finished')


T = TypeVar('T')

class Queue(Generic[T]):
    def __init__(self):
        self.__queue = multiprocessing.Queue()

    def get(self) -> T:
        return self.__queue.get()

    def put(self, item: T):
        self.__queue.put(item)

    def size(self) -> int:
        return self.__queue.qsize()

    def get_all(self) -> List[T]:
        items = []
        try:
            for item in iter(self.__queue.get_nowait, None):
               items.append(item)
        except Empty:
            pass

        return items

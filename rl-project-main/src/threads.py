from contextlib import contextmanager
from typing import Generic, Iterator, TypeVar
from threading import Lock

from tqdm.std import tqdm
from src.models import Model
from src.utils import MultiUpdatableValue, UpdatableValue


T = TypeVar('T')


class ThreadSafeVariable(Generic[T]):
    def __init__(self, initial_value: T) -> None:
        self.__value = initial_value
        self.__lock = Lock()

    @contextmanager
    def lock_and_get(self) -> Iterator[T]:
        self.__lock.acquire(blocking=True)
        try:
            yield self.__value
        finally:
            self.__lock.release()

    def get(self) -> T:
        return self.__value

    def set(self, value: T):
        self.__value = value

    def get_safe(self) -> T:
        self.__lock.acquire(blocking=True)
        value = self.__value
        self.__lock.release()

        return value

    def set_safe(self, value: T) -> None:
        self.__lock.acquire(blocking=True)
        self.__value = value
        self.__lock.release()


class TrainingState:
    def __init__(self, behavior_model: Model, target_model: Model, epsilons: MultiUpdatableValue, progress_bar: tqdm):
        self.__behavior_model = ThreadSafeVariable(behavior_model)
        self.__target_model = ThreadSafeVariable(target_model)
        self.__epsilons = ThreadSafeVariable(epsilons)
        self.__progress_bar = ThreadSafeVariable(progress_bar)
        self.__T = ThreadSafeVariable(0)
        self.__episode = ThreadSafeVariable(0)

    def behavior_model(self):
        return self.__behavior_model

    def target_model(self):
        return self.__target_model

    def epsilons(self):
        return self.__epsilons

    def progress_bar(self):
        return self.__progress_bar

    def T(self):
        return self.__T

    def episode(self):
        return self.__episode

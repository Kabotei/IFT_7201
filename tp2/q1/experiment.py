from utils import create_dirs, create_parent_dirs, load_dict_json, load_dict_ser, save_dict_json, save_dict_ser
import gym
from gym.core import Env
from poutyne.framework.model import Model
from os.path import join
from pathlib import Path


class Experiment:
    def __init__(self, parent_dir: str, name: str):
        self.__name = name
        self.__dir = join(Path.cwd(), parent_dir, 'experiments', name)
        create_dirs(self.__dir)

    def get_name(self) -> str:
        return self.__name

    def get_last_episode(self) -> int:
        path = join(self.__dir, '.last_episode')
        with open(path, 'r') as file:
            return int(file.readline())

    def save_last_episode(self, episode: int):
        path = join(self.__dir, '.last_episode')
        with open(path, 'w') as file:
            return int(file.write(str(episode)))

    def load_params(self) -> dict:
        path = join(self.__dir, 'params')
        return load_dict_json(path)

    def save_params(self, state: dict):
        path = join(self.__dir, 'params')
        save_dict_json(path, data=state)

    def load_state(self, episode: int = None) -> dict:
        if not episode:
            episode = self.get_last_episode()

        print(f"Loading state from episode {episode}")

        path = join(self.__dir, str(episode), 'state')
        return load_dict_ser(path)

    def save_state(self, state: dict, episode: int):
        path = join(self.__dir, str(episode), 'state')
        create_parent_dirs(path)
        save_dict_ser(path, data=state)

    def load_model_state(self, model: Model, id: str, episode: int = None):
        if not episode:
            episode = self.get_last_episode()

        path = join(self.__dir, str(episode), f'model_{id}')
        model.load_weights(path)

    def load_optim_state(self, model: Model, id: str, episode: int = None):
        if not episode:
            episode = self.get_last_episode()

        path = join(self.__dir, str(episode), f'optim_{id}')
        model.load_optimizer_state(path)

    def save_optim_state(self, model: Model, id: str, episode: int):
        path = join(self.__dir, str(episode), f'optim_{id}')
        create_parent_dirs(path)
        model.save_optimizer_state(path)

    def save_model_state(self, model: Model, id: str, episode: int):
        path = join(self.__dir, str(episode), f'model_{id}')
        create_parent_dirs(path)
        model.save_weights(path)

    def save_to_video(self, environment: Env, episode: str = None) -> Env:
        if not episode:
            episode = self.get_last_episode()

        path = join(self.__dir, 'video')
        environment = gym.wrappers.Monitor(environment, directory=path, force=True)

        return environment

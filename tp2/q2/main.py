from typing import List
import gym
from gym.core import Env
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from tqdm import tqdm
from multiprocessing import Process
import pickle
from time import sleep
from os.path import join
from pathlib import Path
from q2.tilecoding import TileCoder
from q2.utils import ExperimentResult, ExperimentResults, ProcessExecutor, Queue
from utils import moving_average


def get_tile_coder(environment):
    return TileCoder(
        environment.observation_space.high,
        environment.observation_space.low,
        num_tilings=8,
        tiling_dim=8,
        max_size=4096,
    )


def set_random_seed(environment, seed):
    environment.seed(seed)
    np.random.seed(seed)


def one_hot_encode(indices: np.ndarray, size: int) -> np.ndarray:
    array = np.zeros(size, dtype=int)
    array[indices] = 1

    return array


def features_array(state: np.ndarray, action: np.ndarray, tile_coder: TileCoder) -> np.ndarray:
    return one_hot_encode(
        indices=tile_coder.phi(state, action),
        size=tile_coder.size)


def choose_action(theta: np.ndarray, state: np.ndarray, actions: List[int], tile_coder: TileCoder) -> int:
    # commpute features
    phis = np.array([features_array(state, action, tile_coder) for action in actions])
    # compute the scalar product <theta, phi>
    action_vals = phis.dot(theta)

    return np.argmax(action_vals)


def train(environment: Env, max_trajectories: int, gamma: float, _lambda: float, lr: float, early_stop: bool = True, verbose: bool = True):
    tile_coder = get_tile_coder(environment)
    all_actions = list(range(environment.action_space.n))

    theta = np.zeros(tile_coder.size, dtype=float)
    Gs = []

    for trajectory_number in range(max_trajectories):
        state = environment.reset()
        action = choose_action(theta=theta, state=state, actions=all_actions, tile_coder=tile_coder)

        G = 0.0
        traces = np.zeros(tile_coder.size, dtype=float)
        trajectory_done = False

        while not trajectory_done:
            next_state, reward, trajectory_done, _ = environment.step(action)
            G += reward

            features_indices = tile_coder.phi(state, action)

            delta = reward - theta[features_indices].sum()
            traces[features_indices] = 1

            if trajectory_done:
                theta += lr * delta * traces
                continue

            next_action = choose_action(theta=theta, state=next_state, actions=all_actions, tile_coder=tile_coder)            

            next_features_indices = tile_coder.phi(next_state, next_action)

            delta += gamma * theta[next_features_indices].sum()
            theta += lr * delta * traces
            traces *= gamma * _lambda
            state = next_state
            action = next_action

        Gs.append(G)

        if verbose and (trajectory_number + 1) % 10 == 0:
            print(f'Episode {trajectory_number+1} : G = {G}')

        if early_stop and len(Gs) >= 100 and np.mean(Gs[-100:]) >= -110:
            break

    return theta, Gs, tile_coder


def test(environment: Env, theta: np.ndarray, tile_coder: TileCoder, n_trajectories: int):
    all_actions = list(range(environment.action_space.n))
    Gs = []

    for _ in range(n_trajectories):
        G = 0.0
        trajectory_done = False
        state = environment.reset()

        while not trajectory_done:
            action = choose_action(theta=theta, state=state, actions=all_actions, tile_coder=tile_coder)            
            next_state, reward, trajectory_done, _ = environment.step(action)
            environment.render()

            G += reward
            state = next_state
        
        Gs.append(G)

    environment.close()
    return Gs


def plot(results: ExperimentResults, fig_path: str):
    plt.figure(1)
    plt.title(f'Performance des agents SARSA($\lambda$) selon différentes valeurs de $\lambda$\n(moyenné sur {results.runs} essais avec moyenne mobile sur 20 épisode)')
    plt.xlabel('Épisode')
    plt.ylabel('Gain')

    X = list(range(results.max_trajectories))

    for result in sorted(results.results, key=lambda x: x._lambda):
        mean_rewards = np.mean(result.rewards, axis=0)
        plt.plot(moving_average(mean_rewards, 20), label=f'$\lambda$ = {result._lambda}')
        
    plt.legend()
    plt.savefig(f'{fig_path}_0.png', format='png')

    plt.figure(2)
    plt.title(f"Performance des agents SARSA($\lambda$) selon différentes valeurs de $\lambda$\n(moyenné sur {results.runs} essais avec écart-type)")
    plt.xlabel('Épisode')
    plt.ylabel('Gain')

    X = list(range(results.max_trajectories))

    for result in sorted(results.results, key=lambda x: x._lambda):
        mean_rewards = np.mean(result.rewards, axis=0)
        std_rewards = np.std(result.rewards, axis=0)

        plt.plot(mean_rewards, label=f'$\lambda$ = {result._lambda}')
        plt.fill_between(X, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
        
    plt.legend()
    plt.savefig(f'{fig_path}_1.png', format='png')

    plt.show()


def parse_args():
    argparser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    argparser.add_argument('mode', choices=['single', 'compare', 'plot'])
    argparser.add_argument('-n', dest='max_trajectories', type=int, default=500, help="max number of trajectories")
    argparser.add_argument('-l', '--lambda', dest='_lambda', type=float, default=0.9, help="Lambda value (for 'single' mode only)")
    argparser.add_argument('-r', '--runs', type=int, default=30, help="Number of runs (for 'figure' mode only)")

    return argparser.parse_args()


class ExperimentProcess(Process):
    def __init__(self, runs: int, max_trajectories: int, _lambda: float, gamma: float, lr: float, queue: Queue, progress_bar: tqdm):
        super().__init__()

        self.__runs = runs
        self.__max_trajectories = max_trajectories
        self.__lambda = _lambda
        self.__gamma = gamma
        self.__lr = lr
        self.__queue = queue
        self.__progress_bar = progress_bar

    def run(self) -> None:
        environment = gym.make("MountainCar-v0")
        all_rewards = []

        for i in range(self.__runs):
            set_random_seed(environment, i)

            _, Gs, _ = train(environment=environment, max_trajectories=self.__max_trajectories, gamma=self.__gamma, _lambda=self.__lambda, lr=self.__lr, early_stop=False, verbose=False)
            
            all_rewards.append(Gs)
            self.__progress_bar.update()

        self.__queue.put(ExperimentResult(
            _lambda=self.__lambda,
            rewards=all_rewards
        ))


def main(args):
    environment = gym.make("MountainCar-v0")
    data_path = join(Path.cwd(), 'q2', 'results')
    fig_path = join(Path.cwd(), 'q2', 'fig')

    learning_rate = 0.1
    gamma = 1.0
    max_trajectories = args.max_trajectories

    if args.mode == 'single':
        set_random_seed(environment, 42)
        _lambda = args._lambda
        theta, Gs, tile_coder = train(environment=environment, max_trajectories=max_trajectories, gamma=gamma, _lambda=_lambda, lr=learning_rate)
        test(environment=environment, n_trajectories=3, theta=theta, tile_coder=tile_coder)
    elif args.mode == 'compare':
        runs = args.runs
        lambdas = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        # lambdas = [0, 0.1, 0.9, 1]
        # lambdas = [0.9]

        processes: List[ExperimentProcess] = []
        progress_bars: List[tqdm] = []
        results: Queue[ExperimentResult] = Queue()

        print(f'Training with lambdas = {lambdas}')

        for _lambda in lambdas:
            progress_bar = tqdm(total=runs)
            progress_bars.append(progress_bar)

            process = ExperimentProcess(
                runs=runs,
                max_trajectories=max_trajectories,
                gamma=gamma,
                lr=learning_rate,
                _lambda=_lambda,
                queue=results,
                progress_bar=progress_bar,
            )
            processes.append(process)

        executor = ProcessExecutor(processes)
        executor.execute()

        while results.size() < len(lambdas):
            sleep(2)

        for progress_bar in progress_bars:
            progress_bar.close()

        experiment_results = ExperimentResults(
            max_trajectories=max_trajectories,
            results=results.get_all()
        )

        with open(data_path, 'wb') as file:
            pickle.dump(experiment_results, file)

        plot(experiment_results, fig_path=fig_path)
    elif args.mode == 'plot':
        with open(data_path, 'rb') as file:
            experiment_results = pickle.load(file)

        plot(experiment_results, fig_path=fig_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)

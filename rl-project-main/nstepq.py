from datetime import datetime
from threading import Thread
from queue import Queue
from typing import List
from dataclasses import dataclass
import numpy as np
import torch
from copy import deepcopy
from torch.functional import Tensor
from torch.nn import MSELoss
from tqdm import tqdm
from src.buffer import Sequence
from src.env import Environment, FixedPatience, FixedReward, Observation, Patience, RewardStrategy, StepsFactorPatience
from src.graph import Graph, GraphFactory
from src.models import create_model
from src.plotting import plot, plot_distance_diffs, plot_distance_diffs_relative, plot_epsilons, plot_gains, plot_losses
from src.threads import TrainingState
from src.transformer import BaseInput, InputTransformer, Normalizer
from src.utils import MultiUpdatableValue, StringBuffer, UpdatableValue


@dataclass
class Result:
    thread_name: str
    gain: float
    loss: float
    distance_diff: float
    optimal_path_steps: int
    epsilon: float


@dataclass
class TrainingParams:
    n_trajectories: int
    reward: RewardStrategy
    patience: Patience
    transformer: InputTransformer
    lr: float
    gamma: float
    epsilons: MultiUpdatableValue
    target_update_interval: int
    tau: float
    new_trajectory_every: int
    min_length: int
    n_steps: int = 5
    seed: int = 42
    save_states: bool = True
    verbose: bool = False


class NSteqQLearningThread(Thread):
    def __init__(self, graph: Graph, params: TrainingParams, training_state: TrainingState, results_queue: Queue, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__graph = graph
        self.__params = params
        self.__training_state = training_state
        self.__results_queue = results_queue

    def run(self) -> None:
        out = StringBuffer()
        env = Environment(
            graph=self.__graph,
            reward=self.__params.reward,
            patience=self.__params.patience,
            new_trajectory_every=self.__params.new_trajectory_every,
            min_length=self.__params.min_length)

        with self.__training_state.episode().lock_and_get() as shared_episode:
            episode = shared_episode + 1
            self.__training_state.episode().set(episode)

        with self.__training_state.T().lock_and_get() as shared_T:
            T = shared_T + 1
            self.__training_state.T().set(T)

        with self.__training_state.behavior_model().lock_and_get() as shared_behavior_model:
            behavior_model = shared_behavior_model.snapshot()
        target_model = deepcopy(self.__training_state.target_model().get_safe())

        should_resync_models = False

        while episode < self.__params.n_trajectories:
            epsilon = self.__training_state.epsilons().get().get()

            if self.__params.verbose:
                out.add(f'Episode {episode}\tepsilon = {epsilon.get():.3f}')

            state, shortest_distance, n_nodes, new_trajectory = env.reset()
            
            if self.__params.verbose:
                out.add(f'Short : {shortest_distance:.2f}\tsteps : {n_nodes}')

            t = 0
            rewards = []
            all_rewards = []
            sequences: List[Sequence] = []
            distance = 0

            while True:
                actions = env.available_actions(state)
                action = behavior_model.choose_action(state, actions, epsilon.get())
                observation: Observation = env.play(action)

                sequences.append(Sequence(state=state, choosen_action=action, reward=observation.reward, next_state=observation.next_state, trajectory_done=None))
                rewards.append(observation.reward)
                distance += observation.distance
                state = observation.next_state
                t += 1

                with self.__training_state.T().lock_and_get() as shared_T:
                    T = shared_T + 1
                    self.__training_state.T().set(T)

                if observation.done or t == self.__params.n_steps:
                    targets = []
                    R = 0 if observation.done else np.max(target_model.predict_all(state, env.available_actions(state)))
                    
                    for r in reversed(rewards):
                        R = r + R*self.__params.gamma
                        targets.append(R)

                    batch_input = list(map(lambda seq: self.__params.transformer.to_net_input(seq.state, seq.choosen_action), reversed(sequences)))
                    batch_input = Tensor(np.array(batch_input))

                    # predicted = behavior_model.predict_on_batch_with_gradients(batch_input)
                    targets = Tensor([list(reversed(targets))]).float().T

                    with self.__training_state.behavior_model().lock_and_get() as shared_behavior_model:
                        # loss = shared_behavior_model.backprop(predicted, targets)
                        loss = shared_behavior_model.train_on_batch(batch_input, targets)

                    all_rewards.extend(rewards)
                    sequences = []
                    rewards = []
                    t = 0
                    should_resync_models = True

                if T % self.__params.target_update_interval == 0:
                    with self.__training_state.target_model().lock_and_get() as shared_target_model:
                        shared_target_model.soft_update_with(behavior_model, tau=self.__params.tau)

                if should_resync_models:
                    should_resync_models = False
                    with self.__training_state.behavior_model().lock_and_get() as shared_behavior_model:
                        behavior_model = shared_behavior_model.snapshot()
                    target_model = deepcopy(self.__training_state.target_model().get_safe())

                if observation.done:
                    break

            with self.__training_state.episode().lock_and_get() as shared_episode:
                episode = shared_episode + 1
                self.__training_state.episode().set(episode)

            distance_diff = distance / shortest_distance - 1

            if not self.__params.verbose:
                with self.__training_state.progress_bar().lock_and_get() as progress_bar:
                    progress_bar.update()

            with self.__training_state.epsilons().lock_and_get():
                epsilon.step()
            
            result = Result(
                thread_name=self.name,
                gain=np.sum(all_rewards),
                loss=loss,#.detach().item(),
                distance_diff=distance_diff,
                optimal_path_steps=n_nodes,
                epsilon=epsilon.get())
            self.__results_queue.put(result)

            if self.__params.verbose:
                out.add(f'Found : {distance:.2f}\tsteps : {len(all_rewards)}\tdiff : {distance_diff:.2f}')
                out.print()


class AsyncNSteqQLearningRunner:
    def __init__(self,
        n_threads: int,
        graph: Graph,
        params: TrainingParams,
        training_state: TrainingState,
        results_queue: Queue
    ) -> None:
        self.__n_threads = n_threads
        self.__graph = graph
        self.__params = params
        self.__training_state = training_state
        self.__results_queue = results_queue

    def start(self):
        threads: List[Thread] = []

        for _ in range(self.__n_threads):
            thread = NSteqQLearningThread(graph=self.__graph, params=self.__params, training_state=self.__training_state, results_queue=self.__results_queue)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()


def train(params: TrainingParams, n_threads: int):
    graph = GraphFactory(seed=params.seed).random(
        n_nodes=20,
        max_node_connections=4,
        min_x=-180,
        max_x=180,
        min_y=-90,
        max_y=90,
        max_weight=1)

    with plot():
        # graph.plot()
        graph.print()

    behavior_model = create_model(
        transformer=params.transformer,
        lr=params.lr, loss_func=MSELoss(),
        in_dim=params.transformer.input_size(), out_dim=1, n_hidden_layers=3, hidden_dim=64,
        seed=params.seed)
    target_model = behavior_model.snapshot()
    progress_bar = tqdm(total=params.n_trajectories)

    state = TrainingState(behavior_model=behavior_model, target_model=target_model, epsilons=params.epsilons, progress_bar=progress_bar)
    results_queue = Queue()
    
    runner = AsyncNSteqQLearningRunner(n_threads=n_threads, graph=graph, params=params, training_state=state, results_queue=results_queue)
    
    runner.start()

    behavior_model = state.behavior_model().get_safe()
    taregt_model = state.target_model().get_safe()
    timestamp = str(datetime.now().timestamp())
    
    if params.save_states:
        behavior_model.save(id='behavior', timestamp=timestamp, with_optim=True)
        taregt_model.save(id='target', timestamp=timestamp, with_optim=False)
        graph.save(timestamp=timestamp)

    results: List[Result] = []
    while not results_queue.empty():
        results.append(results_queue.get())

    epsilons = list(map(lambda r: r.epsilon, results))
    losses = list(map(lambda r: r.loss, results))
    Gs = list(map(lambda r: r.gain, results))
    distance_diffs = list(map(lambda r: r.distance_diff, results))
    optimal_path_steps = list(map(lambda r: r.optimal_path_steps, results))

    with plot():
        plot_epsilons(epsilons)
        plot_losses(losses)
        plot_gains(Gs)
        plot_distance_diffs(distance_diffs)
        plot_distance_diffs_relative(distance_diffs, optimal_path_steps)


def main():
    SEED = 32

    normalizer = Normalizer(mins=[-180, -90], maxs=[180, 90])
    transformer = BaseInput(normalizer=normalizer)

    epsilon_values = [
        UpdatableValue(initial_value=1, min_value=0.1, decay=0.002),
        UpdatableValue(initial_value=1, min_value=0.01, decay=0.005),
        UpdatableValue(initial_value=1, min_value=0.5, decay=0.001),
    ]
    epsilon_probs = [0.4, 0.3, 0.3]
    epsilons = MultiUpdatableValue(epsilon_values, epsilon_probs, seed=SEED)

    params = TrainingParams(
        n_trajectories=5000,
        n_steps=5,
        reward=FixedReward(),
        patience=StepsFactorPatience(10),
        transformer=transformer,
        lr=0.001,
        gamma=0.99,
        epsilons=epsilons,
        target_update_interval=5,
        tau=0.1,
        new_trajectory_every=1,
        min_length=None,
        seed=SEED, # 420 => length=7 for graph with 20 nodes
        save_states = True,
        verbose = True,
    )

    with torch.autograd.set_detect_anomaly(True):
        train(params=params, n_threads=8)


if __name__ == '__main__':
    main()

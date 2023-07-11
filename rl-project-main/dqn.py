from datetime import datetime
from typing import List, NamedTuple
import numpy as np
from copy import deepcopy
from torch import Tensor
from torch.nn.functional import mse_loss
from src.buffer import ReplayBuffer, Sequence
from src.env import Environment, FixedReward, Observation, Patience, RewardStrategy, StepsFactorPatience
from src.graph import GraphFactory
from src.models import Model, create_model
from src.plotting import plot, plot_distance_diffs, plot_distance_diffs_relative, plot_gains, plot_losses
from src.transformer import InputWithActionWeight, Normalizer, InputTransformer


class FormattedBatch(NamedTuple):
    inputs: np.ndarray
    targets: Tensor


def format_batch(env: Environment, transformer: InputTransformer, batch: List[Sequence], target_network: Model, gamma: float) -> FormattedBatch:
    minibatch_columns = zip(*batch)
    states, actions, rewards, next_states, dones = minibatch_columns
    
    next_q_values = []
    for next_state in next_states:
        available_actions = env.available_actions(next_state)
        q_values = target_network.predict_all(next_state, available_actions)
        next_q_values.append(np.max(q_values))

    targets = np.array(rewards) + np.array(gamma) * np.array(next_q_values) * (1 - np.array(dones))
    targets = Tensor(targets).unsqueeze(1).float()

    inputs = np.array([
        transformer.to_net_input(state, action) for state, action in zip(states, actions)
    ])

    return FormattedBatch(inputs=inputs, targets=targets)


def train(
    n_trajectories: int,
    reward: RewardStrategy,
    patience: Patience,
    lr: float,
    gamma: float,
    epsilon_decay: float,
    min_epsilon: float,
    target_update_interval: int,
    tau: float,
    buffer_size: int,
    batch_size: int,
    seed: int = 42,
    save_states: bool = True
):
    graph = GraphFactory(seed=seed).random(
        n_nodes=10,
        max_node_connections=3,
        min_x=-180,
        max_x=180,
        min_y=-90,
        max_y=90,
        max_weight=1,
    )

    normalizer = Normalizer(mins=[-180, -90], maxs=[180, 90])
    transformer = InputWithActionWeight(normalizer=normalizer)

    env = Environment(graph=graph, reward=reward, patience=patience, new_trajectory_every=0)
    model = create_model(
        transformer=transformer,
        lr=lr, loss_func=mse_loss,
        in_dim=transformer.input_size(), out_dim=1, n_hidden_layers=3, hidden_dim=64,
        seed=seed)
    target_model = deepcopy(model)
    replay_buffer = ReplayBuffer(buffer_size=buffer_size)
    epsilon = 1
    Gs = []
    losses = []
    distance_diffs = []
    optimal_path_steps = []

    for trajectory_number in range(n_trajectories):
        print(f'Episode {trajectory_number+1}\tepsilon = {epsilon:.3f}')
        state, shortest_distance, n_nodes = env.reset()
        reached_goal = False
        G = []
        distance = 0

        while not reached_goal:
            actions = env.available_actions(state)
            action = model.choose_action(state, actions, epsilon)
            observation: Observation = env.play(action)

            G.append(observation.reward)
            distance += observation.distance

            reached_goal = observation.done
            state = observation.next_state

            replay_buffer.store(Sequence(
                state=state,
                choosen_action=action,
                reward=observation.reward,
                next_state=observation.next_state,
                trajectory_done=observation.done
            ))

            if len(replay_buffer) >= batch_size:
                minibatch = replay_buffer.get_batch(batch_size)
                formatted_batch: FormattedBatch = format_batch(env=env, transformer=transformer, batch=minibatch, target_network=target_model, gamma=gamma)
                loss = model.train_on_batch(formatted_batch.inputs, formatted_batch.targets)
                losses.append(loss)

                if trajectory_number % target_update_interval == 0:
                    target_model.soft_update_with(model, tau=tau)

        distance_diff = distance / shortest_distance - 1
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        
        Gs.append(np.sum(G))
        distance_diffs.append(distance_diff)
        optimal_path_steps.append(n_nodes)
        
        print(f'diff : {distance_diff:.1f}\tsteps : {len(G)}\tgain : {G[-1]:.2f}\tsum : {np.sum(G):.2f}\n')

    if save_states:
        timestamp = str(datetime.now().timestamp())
        model.save(id='base', timestamp=timestamp)
        target_model.save(id='target', timestamp=timestamp)
        graph.save(timestamp=timestamp)

    with plot():
        plot_losses(losses)
        plot_gains(Gs)
        plot_distance_diffs(distance_diffs, optimal_path_steps)
        plot_distance_diffs_relative(distance_diffs, optimal_path_steps)


if __name__ == '__main__':
    train(
        n_trajectories=500,
        reward=FixedReward(),
        patience=StepsFactorPatience(10),
        lr=0.0001,
        gamma=0.99,
        epsilon_decay=0.99,
        min_epsilon=0.01,
        target_update_interval=4,
        tau=0.01,
        buffer_size=10000,
        batch_size=64,
        seed=420,
        save_states = False
    )

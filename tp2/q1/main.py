import random
from typing import List, Tuple
from gym.core import Env
from poutyne import Model
from copy import deepcopy  # NEW
import numpy as np
import gym
import torch
from torch import Tensor
from torch.nn.functional import mse_loss
from matplotlib import pyplot as plt
from q1.experiment import Experiment
from q1.types import Sequence, TrainingParams, TrainingState
from q1.utils import parse_args, plot_losses, plot_rewards, save_training_state, setup_experiment


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


def format_batch(batch: List[Sequence], target_network: Model, gamma: float):
    '''
    Input : 
        - batch, a list of n=batch_size elements from the replay buffer
        - target_network, the target network to compute the one-step lookahead target
        - gamma, the discount factor

    Returns :
        - states, a numpy array of size (batch_size, state_dim) containing the states in the batch
        - (actions, targets) : where actions and targets both
                      have the shape (batch_size, ). Actions are the 
                      selected actions according to the target network
                      and targets are the one-step lookahead targets.
    '''
    minibatch_columns = map(lambda x: np.array(x), zip(*batch))
    
    states, actions, rewards, next_states, trajectories_done = minibatch_columns

    next_q_values = target_network.predict_on_batch(next_states)
    targets = get_targets(next_q_values, rewards, trajectories_done, gamma).astype(np.float32)

    return states, (actions, targets)


def dqn_loss(y_pred: Tensor, y_target: Tuple[Tensor, Tensor]) -> Tensor:
    '''
    Input :
        - y_pred, (batch_size, n_actions) Tensor outputted by the network
        - y_target = (actions, targets), where actions and targets both
                      have the shape (batch_size, ). Actions are the 
                      selected actions according to the target network
                      and targets are the one-step lookahead targets.

    Returns :
        - The DQN loss 
    '''
    actions, Q_target = y_target
    Q_predict = y_pred.gather(1, actions.unsqueeze(-1)).squeeze()

    return mse_loss(Q_predict, Q_target)


def get_targets(next_q_vals, rewards, terminal, gamma):
    '''
    Returns Q-Learning targets according to the 1-step SARSA lookahead formula,
    i.e. target_t = r_t + gamma * max(Q(s_t+1))

    If s_t was already terminal, then we only have target_t = r_t.

    next_q_vals: (batch_size, 2) numpy array representing the Q(s_t+1) values
    rewards: (batch_size,) numpy array representing the r_t values
    terminal: (batch_size,) boolean numpy array representing if s_t+1 was terminal
    gamma: float between 0 and 1

    Returns a (batch_size,) numpy array containing the 1-step lookahead targets.
    '''
    
    next_action_values_selected = np.max(next_q_vals, axis=-1)
    targets = rewards + gamma * next_action_values_selected * (1 - terminal)

    return targets


def set_random_seed(environment: Env, seed: int = 42):
    environment.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(experiment: Experiment, params: TrainingParams, training_state: TrainingState, environment: Env, model: Model, target_model: Model):
    set_random_seed(environment, params.seed)

    replay_buffer = ReplayBuffer(
        buffer_size=params.buffer_size,
        data=training_state.replay_buffer
    )
    losses = training_state.losses
    epsilon = training_state.epsilon
    initial_trajectory_number = training_state.trajectory_number
    all_rewards = training_state.rewards

    for trajectory_number in range(initial_trajectory_number, params.n_trajectories):
        trajectory_done = False
        G = 0
        state = environment.reset().astype(np.float32)

        while not trajectory_done:
            action = model.get_action(state, epsilon)
            next_state, reward, trajectory_done, _ = environment.step(action)
            next_state = next_state.astype(np.float32)

            G += reward
            replay_buffer.store(Sequence(state, action, reward, next_state, trajectory_done))

            state = next_state

            if len(replay_buffer) > params.batch_size:
                minibatch = replay_buffer.get_batch(params.batch_size)
                states, (actions_taken, targets) = format_batch(batch=minibatch, target_network=target_model, gamma=params.gamma)
                loss = model.train_on_batch(states, (actions_taken, targets))
                losses.append(loss)

                if trajectory_number % params.training_interval == 0:
                    target_model.soft_update(model, params.tau)

        all_rewards.append(G)
        epsilon = max(epsilon * params.epsilon_decay, params.epsilon_min)

        if (trajectory_number + 1) % 5 == 0:
            print(f"Episode {trajectory_number + 1} : G = {G:.2f}, epsilon = {epsilon:4f}")

        if (trajectory_number + 1) % params.saving_interval == 0:
            episode = trajectory_number + 1
            training_state = TrainingState(
                trajectory_number=episode,
                epsilon=epsilon,
                rewards=all_rewards,
                losses=losses,
                replay_buffer=replay_buffer.get_list()
            )
            save_training_state(
                experiment=experiment,
                state=training_state,
                model=model,
                target_model=target_model,
                episode=episode
            )

    return all_rewards, losses


def test(params: TrainingParams, training_state: TrainingState, environment: Env, model: Model):
    set_random_seed(environment, params.seed)

    epsilon = training_state.epsilon
    n_trajectories = params.n_trajectories

    for _ in range(n_trajectories):
        trajectory_done = False
        state = environment.reset().astype(np.float32)

        while not trajectory_done:
            environment.render()

            action = model.get_action(state, epsilon)
            next_state, _, trajectory_done, _ = environment.step(action)
            next_state = next_state.astype(np.float32)

            state = next_state

    environment.close()


def main(mode: str, experiment_name: str = None, episode: int = None, n_trajectories: int = None, save: bool = False):
    '''
    All hyperparameter values and overall code structure are only given as a baseline. 
    
    You can use them if they help  you, but feel free to implement from scratch the
    required algorithms if you wish!
    '''
    environment = gym.make("LunarLander-v2")

    setup = setup_experiment(
        environment=environment,
        loss_func=dqn_loss,
        experiment_name=experiment_name,
        episode=episode,
        n_trajectories=n_trajectories,
    )

    if mode == 'train':
        print('Starting training')
        rewards, losses = train(experiment=setup.experiment, params=setup.params, training_state=setup.state, environment=environment, model=setup.model, target_model=setup.target_model)
        plot_rewards(rewards)
        plot_losses(losses)
        plt.show()
    elif mode == 'test':
        if save:
            environment = setup.experiment.save_to_video(environment=environment, episode=episode)
        test(params=setup.params, training_state=setup.state, environment=environment, model=setup.model)


if __name__ == "__main__":
    args = parse_args()
    main(
        mode=args.mode,
        experiment_name=args.experiment,
        n_trajectories=args.n_trajectories,
        episode=args.episode,
        save=args.save
    )

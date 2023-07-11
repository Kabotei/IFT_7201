from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
from typing import List
from gym.core import Env
from poutyne import Model
from dataclasses import asdict
from torch.optim.adam import Adam
from matplotlib import pyplot as plt
from q1.experiment import Experiment
from q1.models import DQN, NNModel
from q1.types import ExperimentSetup, Models, TrainingParams, TrainingState
from utils import moving_average


def plot_rewards(rewards: List[float]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Gains par épisode (+ moyenne mobile sur 20 essais)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Gains')
    ax.axhline(0, color='black')
    ax.plot(rewards, label='Gains')
    ax.plot(moving_average(rewards, 20), label='Gains (MM 20 essais)')
    ax.legend()


def plot_losses(losses: List[float]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Perte par mise à jour (moyenne mobile sur 100 essais)')
    ax.set_xlabel('Mise à jour')
    ax.set_ylabel('Perte')
    ax.plot(moving_average(losses, 100))


def save_training_state(experiment: Experiment, state: TrainingState, model: Model, target_model: Model, episode: int):
    experiment.save_state(asdict(state), episode=episode)
    experiment.save_model_state(model, episode=episode, id='model')
    experiment.save_optim_state(model, episode=episode, id='model')
    experiment.save_model_state(target_model, episode=episode, id='target_model')
    experiment.save_last_episode(episode=episode)
    print(f"Saved state for episode {episode}")


def init_models(environment: Env, lr: float, loss_func) -> Models:
    network = NNModel(
        in_dim=environment.observation_space.shape[0],
        out_dim=environment.action_space.n
    )
    optimizer = Adam(network.parameters(), lr=lr)
    model = DQN(
        actions=environment.action_space,
        network=network,
        optimizer=optimizer,
        loss_function=loss_func
    )

    target_network = NNModel(
        in_dim=environment.observation_space.shape[0],
        out_dim=environment.action_space.n
    )
    target_model = DQN(
        actions=environment.action_space,
        network=target_network,
        optimizer=None,
        loss_function=loss_func
    )
    target_model.set_weights(model.get_weight_copies())

    return Models(model=model, target_model=target_model)


def setup_experiment(environment: Env, loss_func, experiment_name: str = None, episode: int = None, n_trajectories: int = None) -> ExperimentSetup:
    if experiment_name:
        print(f'Loading experiment {experiment_name}')
        experiment = Experiment(parent_dir='q1', name=experiment_name)
        params = TrainingParams(**experiment.load_params())
        params.n_trajectories = n_trajectories
        state = TrainingState(**experiment.load_state(episode=episode))
    else:
        experiment = Experiment(parent_dir='q1', name=str(datetime.now().timestamp()))
        params = TrainingParams(
            n_trajectories = n_trajectories,
        )
        print(f'Creating experiment {experiment.get_name()}')

        state = TrainingState()
        experiment.save_params(asdict(params))

    if n_trajectories:
        params.n_trajectories = n_trajectories

    model, target_model = init_models(environment=environment, lr=params.learning_rate, loss_func=loss_func)
    
    if experiment_name:
        experiment.load_model_state(model, id='model', episode=episode)
        experiment.load_optim_state(model, id='model', episode=episode)
        experiment.load_model_state(target_model, id='target_model', episode=episode)

    return ExperimentSetup(
        experiment=experiment,
        params=params,
        state=state,
        model=model,
        target_model=target_model
    )


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, choices=['train', 'test'])
    parser.add_argument('-e', '--experiment', type=str, help='Experiment to load (ex: 1637016889.028866)')
    parser.add_argument('-n', '--n_trajectories', type=int, help='Number of episodes (trajectories) to run')
    parser.add_argument('--episode', type=int, help='Episode to load')
    parser.add_argument('--save', action='store_true', help='Save video (test mode only)')

    return parser.parse_args()

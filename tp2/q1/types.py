from typing import List, NamedTuple
from poutyne import Model
import numpy as np
from dataclasses import dataclass, field
from q1.experiment import Experiment


class Models(NamedTuple):
    model: Model
    target_model: Model


class Sequence(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    trajectory_done: bool


@dataclass
class TrainingParams:
    batch_size: int = 64
    gamma: float = 0.99
    buffer_size: int = 1e5
    seed: int = 42
    tau: float = 2e-3 # 1e-3
    training_interval: int = 4
    learning_rate: float = 2e-4 # 1e-4
    epsilon_decay: float = 0.99
    epsilon_min: float = 0.01
    n_trajectories: int = 300
    saving_interval: int = 20


@dataclass
class TrainingState:
    trajectory_number: int = 0
    epsilon: float = 1.0
    losses: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    replay_buffer: List[Sequence] = field(default_factory=list)


class ExperimentSetup(NamedTuple):
    experiment: Experiment
    params: TrainingParams
    state: TrainingState
    model: Model
    target_model: Model

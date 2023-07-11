from __future__ import annotations
from copy import deepcopy
from typing import List, Tuple
import poutyne
import numpy as np
import torch
from pathlib import Path
import os
from torch import Tensor
from torch.optim.optimizer import Optimizer
from src.env import Action, State
from src.graph import Node
from src.transformer import InputTransformer


def soft_update(origin_params: np.ndarray, target_params: np.ndarray, factor: float) -> np.ndarray:
    for k in origin_params:
        target_params[k] = (1 - factor) * origin_params[k] + factor * target_params[k]

    return origin_params


class Model(poutyne.Model):
    def __init__(self, transformer: InputTransformer, *args, seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.__transformer = transformer
        self.__seed = seed
        self.__random = np.random.RandomState(seed=seed)

    def snapshot(self) -> Model:
        return Model(transformer=self.__transformer, optimizer=None, network=deepcopy(self.network), loss_function=self.loss_function, seed=self.__seed)

    def backprop(self, predicted: Tensor, targets: Tensor) -> float:
        self.optimizer.zero_grad()

        loss = self.loss_function(predicted, targets)
        loss.backward()

        self.optimizer.step()

        return loss

    def choose_action(self, state: State, actions: List[Node], epsilon: float) -> Action:
        if len(actions) == 1:
            return actions[0]
        elif np.random.rand() < epsilon:
            return self.choose_action_random(actions=actions)
        else:
            return self.choose_action_policy(state=state, actions=actions)

    def choose_action_random(self, actions: List[Node]) -> Action:
        index = self.__random.randint(0, len(actions))
        return actions[index]

    def choose_action_policy(self, state: State, actions: List[Node]) -> Action:
        q_vals = self.predict_all(state, actions)
        index = np.argmax(q_vals)
        return actions[index]

    def predict_all(self, state: State, actions: List[Node]) -> np.ndarray:
        net_input = np.array([
            self.__transformer.to_net_input(state, action) for action in actions
        ])
        q_vals = self.predict_on_batch(net_input)

        return np.array(q_vals)

    def predict_on_batch_with_gradients(self, x) -> Tensor:
        with self._set_training_mode(True):
            x = self.preprocess_input(x)
            return self.network(*x)

    def soft_update_with(self, other: poutyne.Model, tau: float):
        new_weights = soft_update(self.get_weight_copies(), other.get_weight_copies(), tau)
        self.set_weights(new_weights)

    def save(self, id: str, timestamp: str, with_optim: bool = False):
        base_path = os.path.join(Path.cwd(), 'saved', timestamp)
        os.makedirs(base_path, exist_ok=True)

        with open(os.path.join(base_path, f'model_{id}'), 'wb') as file:
            self.save_weights(file)

        if with_optim:
            with open(os.path.join(base_path, f'model_{id}_optim'), 'wb') as file:
                self.save_optimizer_state(file)

    def load(self, id: str, timestamp: str, with_optim: bool = False):
        base_path = os.path.join(Path.cwd(), 'saved', timestamp)

        with open(os.path.join(base_path, f'model_{id}'), 'rb') as file:
            self.load_weights(file)

        if with_optim:
            with open(os.path.join(base_path, f'model_{id}_optim'), 'rb') as file:
                self.load_optimizer_state(file)


class BaseNet(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_hidden_layers: int = 3, hidden_dim: int = 64):
        super().__init__()

        layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                # torch.nn.Dropout(p=0.2),
            ])
        layers.append(torch.nn.Linear(hidden_dim, out_dim))

        self.fc = torch.nn.Sequential(*layers).float()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x.float())


def create_model(transformer: InputTransformer, lr: float, loss_func, in_dim: int, out_dim: int, n_hidden_layers: int, hidden_dim: int, seed: int = 42) -> Model:
    network = BaseNet(in_dim=in_dim, out_dim=out_dim, n_hidden_layers=n_hidden_layers, hidden_dim=hidden_dim)
    # optimizer = torch.optim.Adam(params=network.parameters(), lr=lr)
    optimizer = torch.optim.RMSprop(params=network.parameters(), lr=lr, alpha=0.99, centered=False)
    
    model = Model(
        transformer=transformer,
        network=network,
        optimizer=optimizer,
        loss_function=loss_func,
        seed=seed
    )
    
    return model


def create_model_without_optim(transformer: InputTransformer, loss_func, in_dim: int, out_dim: int, n_hidden_layers: int, hidden_dim: int, seed: int = 42) -> Model:
    network = BaseNet(in_dim=in_dim, out_dim=out_dim, n_hidden_layers=n_hidden_layers, hidden_dim=hidden_dim)
    model = Model(
        transformer=transformer,
        network=network,
        optimizer=None,
        loss_function=loss_func,
        seed=seed
    )
    
    return model


def create_model_using_optim(transformer: InputTransformer, loss_func, in_dim: int, out_dim: int, n_hidden_layers: int, hidden_dim: int, optimizer: Optimizer, seed: int = 42) -> Model:
    network = BaseNet(in_dim=in_dim, out_dim=out_dim, n_hidden_layers=n_hidden_layers, hidden_dim=hidden_dim)
    model = Model(
        transformer=transformer,
        network=network,
        optimizer=optimizer,
        loss_function=loss_func,
        seed=seed
    )
    
    return model

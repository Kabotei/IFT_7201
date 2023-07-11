from torch.nn.functional import mse_loss
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from src.plotting import plot
from src.env import Environment, State
from src.graph import GraphFactory
from src.models import create_model_without_optim
from src.transformer import BaseInput, Normalizer


# timestamp = 'first'
timestamp = '1639745594.15811'
seed = 42

graph = GraphFactory(seed=seed).empty()
graph.load(timestamp=timestamp)

normalizer = Normalizer(mins=[-180, -90], maxs=[180, 90])
transformer = BaseInput(normalizer=normalizer)

env = Environment(graph=graph, reward=None, patience=None)
model = create_model_without_optim(
    transformer=transformer,
    loss_func=mse_loss,
    in_dim=transformer.input_size(), out_dim=1, n_hidden_layers=3, hidden_dim=64,
    seed=seed)
model.load(id='behavior', timestamp=timestamp, with_optim=False)

number_of_tries = 500

alg_method = []
net_method = []
net_method2 = []

for _ in range(number_of_tries):
    state, _, n_nodes, _ = env.reset()

    start = datetime.now()
    graph.shortest_path(state.current_node, state.end_node)
    end = datetime.now()

    alg_method.append((end - start).total_seconds() / n_nodes)

for _ in range(number_of_tries):
    state, _, n_nodes, _ = env.reset()
    start = datetime.now()

    while state.current_node != state.end_node:
        actions = env.available_actions(state)
        q_vals = model.predict_all(state, actions)
        next_node = actions[np.argmax(q_vals)]
        state = State(previous_node=state.current_node, current_node=next_node, end_node=state.end_node)

    end = datetime.now()
    net_method.append((end - start).total_seconds() / n_nodes)

for _ in range(number_of_tries):
    state, _, n_nodes, _ = env.reset()
    dts = []

    while state.current_node != state.end_node:
        start = datetime.now()
        actions = env.available_actions(state)
        q_vals = model.predict_all(state, actions)
        next_node = actions[np.argmax(q_vals)]
        state = State(previous_node=state.current_node, current_node=next_node, end_node=state.end_node)
        end = datetime.now()
        dts.append((end - start).total_seconds())

    net_method2.append(np.mean(dts))


with plot():
    plt.figure()
    plt.title('Time per node')
    plt.xlabel('Tries')
    plt.ylabel('Time')
    plt.plot(alg_method, label='alg')
    plt.plot(net_method, label='net (relative to optimal)')
    plt.plot(net_method2, label='net (relative to found)')
    plt.legend()

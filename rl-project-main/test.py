from torch.nn.functional import mse_loss
from dataclasses import dataclass

from tqdm.std import tqdm
from src.plotting import plot, plot_distance_diffs_relative
from src.env import Environment, FixedPatience, NullReward
from src.graph import GraphFactory
from src.models import create_model_without_optim
from src.transformer import BaseInput, InputTransformer, Normalizer

@dataclass
class TestParams:
    timestamp: str
    n_episodes: int
    transformer: InputTransformer
    seed: int = 42
    verbose: bool = False


def test(params: TestParams):
    graph = GraphFactory(seed=params.seed).empty()
    graph.load(timestamp=params.timestamp)

    env = Environment(graph=graph, reward=NullReward(), patience=FixedPatience(200))
    model = create_model_without_optim(
        transformer=params.transformer,
        loss_func=mse_loss,
        in_dim=params.transformer.input_size(), out_dim=1, n_hidden_layers=3, hidden_dim=64,
        seed=params.seed)
    model.load(id='behavior', timestamp=params.timestamp, with_optim=False)

    distance_diffs = []
    optimal_path_steps = []

    iterable = range(params.n_episodes) if params.verbose else tqdm(range(params.n_episodes))

    for episode in iterable:
        state, shortest_distance, n_nodes, _ = env.reset()
        distance = 0
        should_stop = False
        t = 0

        while not should_stop:
            actions = env.available_actions(state)
            action = model.choose_action_policy(state, actions)
            observation = env.play(action)

            state = observation.next_state
            should_stop = observation.done
            distance += observation.distance
            t += 1

        distance_diff = distance / shortest_distance - 1
        distance_diffs.append(distance_diff)
        optimal_path_steps.append(n_nodes)

        if params.verbose:
            print(f'Episode {episode}')
            print(f'Short : {shortest_distance:.2f}\tsteps : {n_nodes}')
            print(f'Found : {distance:.2f}\tsteps : {t}\tdiff : {distance_diff:.2f}\n')


    with plot():
        plot_distance_diffs_relative(distance_diffs=distance_diffs, n_nodes=optimal_path_steps, type='test', max=None)
        plot_distance_diffs_relative(distance_diffs=distance_diffs, n_nodes=optimal_path_steps, type='test, zoomed', max=10)
        plot_distance_diffs_relative(distance_diffs=distance_diffs, n_nodes=optimal_path_steps, type='test, zoomed x5', max=2)


def main():
    normalizer = Normalizer(mins=[-180, -90], maxs=[180, 90])
    transformer = BaseInput(normalizer=normalizer)

    params = TestParams(
        timestamp='last_model',
        n_episodes=5000,
        transformer=transformer,
        seed=32,
        verbose=True,
    )
    test(params)



if __name__ == '__main__':
    main()

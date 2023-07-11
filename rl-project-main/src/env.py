from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Tuple
import numpy as np
from src.graph import Graph, NoPathFound, Node, Path


class Trajectory(NamedTuple):
    start_node: Node
    end_node: Node


class State(NamedTuple):
    previous_node: Node
    current_node: Node
    end_node: Node

    def reached_goal(self):
        return self.current_node == self.end_node


class Action(NamedTuple):
    next_node: Node
    weight: float

    def to_numpy(self, include_weight: bool = False):
        if include_weight:
            return np.array([*self.next_node.to_numpy(), self.weight])
        else:
            return self.next_node.to_numpy()


class Observation(NamedTuple):
    next_state: State
    reward: float
    done: bool
    distance: float


class RewardStrategy(ABC):
    @abstractmethod
    def calculate(self, state: State, graph: Graph, best_path: Path, choosen_action: Action, available_actions: List[Action], visits: Dict[Node, int], distances: List[float], reached_goal: bool, failed: bool) -> float:
        raise NotImplementedError()


class NullReward(RewardStrategy):
    def calculate(self, state: State, graph: Graph, best_path: Path, choosen_action: Action, available_actions: List[Action], visits: Dict[Node, int], distances: List[float], reached_goal: bool, failed: bool) -> float:
        return 0


class WeightReward(RewardStrategy):
    def calculate(self, state: State, graph: Graph, best_path: Path, choosen_action: Action, available_actions: List[Action], visits: Dict[Node, int], distances: List[float], reached_goal: bool, failed: bool) -> float:
        return -distances[-1]


class FixedReward(RewardStrategy):
    def __init__(self, tolerance: float = 0.1) -> None:
        super().__init__()
        self.__tolerance = tolerance

    def calculate(self, state: State, graph: Graph, best_path: Path, choosen_action: Action, available_actions: List[Action], visits: Dict[Node, int], distances: List[float], reached_goal: bool, failed: bool) -> float:
        if reached_goal:
            return best_path.distance / np.sum(distances)
        elif failed:
            return 0
        else:
            return 0


class Patience(ABC):
    @abstractmethod
    def should_stop(self, best_path: Path, distances: List[float]) -> bool:
        raise NotImplementedError()


class NullPatience(Patience):
    def should_stop(self, best_path: Path, distances: List[float]) -> bool:
        return False


class FixedPatience(Patience):
    def __init__(self, steps: int) -> None:
        self.__steps = steps

    def should_stop(self, best_path: Path, distances: List[float]) -> bool:
        return len(distances) >= self.__steps


class StepsFactorPatience(Patience):
    def __init__(self, factor: int):
        self.__factor = factor

    def should_stop(self, best_path: Path, distances: List[float]) -> bool:
        return len(distances) >= self.__factor * best_path.n


class DistanceFactorPatience(Patience):
    def __init__(self, factor: int):
        self.__factor = factor

    def should_stop(self, best_path: Path, distances: List[float]) -> bool:
        return np.sum(distances) >= self.__factor * best_path.distance


class Environment:
    def __init__(self, graph: Graph, reward: RewardStrategy, patience: Patience, new_trajectory_every: int = 1, min_length: int = None):
        self.__graph = graph
        self.__patience = patience
        self.__reward = reward
        self.__new_trajectory_every = new_trajectory_every
        self.__min_length = min_length

        self.__trajectories: int = 0
        self.__trajectory: Trajectory = None

        self.__state: State = None
        self.__best_path: Path = None

        self.__distances: List[float] = []
        self.__current_available_actions: List[Action] = []
        self.__visits: Dict[Node, int] = {}

    def available_actions(self, state: State = None) -> List[Action]:
        if state == self.__state:
            return self.__current_available_actions

        if not state:
            state = self.__state

        next_edges = self.__graph.edges_from(state.current_node)
        if state.previous_node is not None:
            to_remove = filter(lambda edge: edge.to_node == state.previous_node, next_edges)
            next_edges.remove(to_remove)

        return list(map(lambda edge: Action(next_node=edge.to_node, weight=edge.weight), next_edges))

    def play(self, action: Action) -> Observation:
        if self.__should_stop():
            raise RuntimeError("End node has already been reached. Please reset the environment.")

        if action not in self.__current_available_actions:
            raise ValueError(f"Action '{action}' is unavailable for current state. Please choose from available ones.")

        self.__distances.append(action.weight)

        self.__state = State(
            previous_node = self.__state.previous_node,
            current_node=action.next_node,
            end_node=self.__state.end_node,
        )
        self.__increment_visits(action.next_node)
        self.__current_available_actions = self.available_actions()

        reward = self.__reward.calculate(
            state=self.__state,
            graph=self.__graph,
            best_path=self.__best_path,
            choosen_action = action,
            available_actions = self.__current_available_actions,
            visits = self.__visits,
            distances=self.__distances,
            reached_goal=self.__reached_goal(),
            failed=self.__failed(),
        )

        return Observation(
            next_state=self.__state,
            reward=reward,
            done=self.__should_stop(),
            distance=action.weight,
        )

    def __increment_visits(self, node: Node):
        self.__visits[node] = self.__visits.get(node, 0) + 1

    def __should_stop(self) -> bool:
        return self.__reached_goal() or self.__failed()

    def __reached_goal(self) -> bool:
        return self.__state.reached_goal()

    def __failed(self) -> bool:
        return self.__patience.should_stop(self.__best_path, self.__distances)

    def reset(self) -> Tuple[State, float, int, bool]:
        self.__distances = []
        new_trajectory = False

        if not self.__trajectory:
            self.__best_path, self.__trajectory = self.__find_new_trajectory()
            new_trajectory = True

        if self.__new_trajectory_every != 0 and self.__trajectories % self.__new_trajectory_every == 0:
            self.__best_path, self.__trajectory = self.__find_new_trajectory()
            self.__trajectories = 0
            new_trajectory = True

        self.__state = State(
            previous_node=None,
            current_node=self.__trajectory.start_node,
            end_node=self.__trajectory.end_node,
        )
        self.__current_available_actions = self.available_actions()
        self.__trajectories += 1

        return self.__state, self.__best_path.distance, self.__best_path.n, new_trajectory

    def __find_new_trajectory(self) -> Tuple[Path, Trajectory]:
        best_path: Path = None
        start_node = self.__graph.random_node()
        end_node = self.__graph.random_node()

        while best_path == None or start_node == end_node or (self.__min_length and best_path.n <= self.__min_length):
            try:
                best_path = self.__graph.shortest_path(start_node, end_node)
            except NoPathFound:
                start_node = self.__graph.random_node()
                end_node = self.__graph.random_node()

        return best_path, Trajectory(start_node=start_node, end_node=end_node)

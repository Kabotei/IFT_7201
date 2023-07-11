from threading import Lock
from typing import List, NamedTuple, Optional
from matplotlib.collections import LineCollection
import numpy as np
from graph import BasicGraph, shortest_path
from graph.visuals import plot_2d
from scipy import spatial
import os
import pickle
import pathlib
from matplotlib import pyplot as plt


class Node(NamedTuple):
    x: float
    y: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y])


class Edge(NamedTuple):
    from_node: Node
    to_node: Node
    weight: float


class Path(NamedTuple):
    nodes: List[Node]
    distance: float
    n: int


class NoPathFound(RuntimeError):
    def __init__(self, from_node: Node, to_node: Node):
        super().__init__(f'Could not find a path from node {from_node} to node {to_node}.')


class Graph:
    def __init__(self, graph: BasicGraph, seed: int = 42):
        self.__random = np.random.RandomState(seed=seed)
        self.__graph = graph
        self.__lock = Lock()

    def __str__(self) -> str:
        return str(self.__graph)

    def __repr__(self) -> str:
        return self.__str__()

    def random_node(self) -> Node:
        all_nodes = self.__graph.nodes()

        self.__lock.acquire(blocking=True)
        node_index = self.__random.randint(0, len(all_nodes))
        self.__lock.release()

        return Node(*all_nodes[node_index])

    def next_nodes(self, from_node: Node) -> List[Node]:
        return self.__graph.nodes(from_node=from_node)

    def weight_between(self, from_node: Node, to_node: Node) -> Optional[float]:
        return self.__graph.edge(from_node, to_node, None)

    def edges_from(self, node: Node) -> List[Edge]:
        return [
            Edge(from_node=edge[0], to_node=edge[1], weight=edge[2])
            for edge in self.__graph.edges(from_node=node)
        ]
    
    def edge_between(self, from_node: Node, to_node: Node):
        weight = self.__graph.edge(from_node, to_node, 0)
        return Edge(from_node=from_node, to_node=to_node, weight=weight)

    def shortest_path(self, from_node: Node, to_node: Node) -> Path:
        if from_node == to_node:
            raise NoPathFound(from_node, to_node)

        path = shortest_path(self.__graph, start=from_node, end=to_node)
        
        if path == (float("inf"), []):
            raise NoPathFound(from_node, to_node)

        return Path(
            nodes=path[1],
            distance=path[0],
            n=len(path[1]) - 1
        )

    def mutate_nodes(self, factor: float):
        pass

    def mutate_edges(self, factor: float):
        pass

    def plot(self):
        plot_2d(self.__graph)

    def print(self):
        lines = np.array([[edge[0], edge[1]] for edge in self.__graph.edges()])
        values = [edge[2] for edge in self.__graph.edges()]
        nodes = np.array(self.__graph.nodes())

        fig, ax = plt.subplots()
        
        lines = LineCollection(lines, array=values, cmap=plt.cm.get_cmap('viridis'), linewidths=2)
        ax.add_collection(lines)
        ax.scatter(nodes[:, 0], nodes[:, 1], color='#1874ba')
        
        fig.colorbar(lines, label='weight')
        ax.autoscale()

        ax.set_xlabel('x')
        ax.set_ylabel('y')


    def save(self, timestamp: str):
        base_path = os.path.join(pathlib.Path.cwd(), 'saved', timestamp)
        os.makedirs(base_path, exist_ok=True)

        with open(os.path.join(base_path, 'graph'), 'wb') as file:
            pickle.dump(self.__graph, file)

    def load(self, timestamp: str):
        base_path = os.path.join(pathlib.Path.cwd(), 'saved', timestamp)

        with open(os.path.join(base_path, 'graph'), 'rb') as file:
            self.__graph = pickle.load(file)


class GraphFactory:
    def __init__(self, seed: int = 42):
        self.__seed = seed
        self.__random = np.random.RandomState(seed=seed)

    def empty(self):
        return Graph(graph=BasicGraph(), seed=self.__seed)

    def from_edges(self, edges: List[Edge]) -> Graph:
        graph = BasicGraph()
        
        for edge in edges:
            graph.add_edge(edge.from_node, edge.to_node, value=edge.weight, bidirectional=True)

        return Graph(graph=graph, seed=self.__seed)

    def random(self,
        n_nodes: int,
        max_node_connections: int,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        max_weight: float
    ) -> Graph:
        if max_node_connections < 2:
            raise ValueError('max_node_connections should be at least 2')

        if max_node_connections >= n_nodes:
            raise ValueError('max_node_connections should be less than n_nodes')
        
        graph = BasicGraph()
        

        Xs = self.__random.uniform(min_x, max_x, size=n_nodes)
        Ys = self.__random.uniform(min_y, max_y, size=n_nodes)
        coords = np.array([Xs, Ys]).T
        coords_tree = spatial.KDTree(coords)

        for from_coord_index, from_coord in enumerate(coords):
            from_node = Node(x=from_coord[0], y=from_coord[1])

            n_connected_nodes = graph.out_degree(from_node)
            if n_connected_nodes == max_node_connections:
                continue

            max_n_new_neighbours = max_node_connections - n_connected_nodes
            nearest_coords = coords_tree.query(from_coord, k=max_n_new_neighbours+1)[1]

            for to_coord_index in nearest_coords:
                if to_coord_index == from_coord_index:
                    continue
                
                to_coord = coords[to_coord_index]
                to_node = Node(x=to_coord[0], y=to_coord[1])

                n_connected_nodes = graph.out_degree(to_node)
                if n_connected_nodes == max_node_connections:
                    break

                weight = self.__random.uniform(0, max_weight)
                graph.add_edge(from_node, to_node, value=weight, bidirectional=True)

        return Graph(graph=graph, seed=self.__seed)

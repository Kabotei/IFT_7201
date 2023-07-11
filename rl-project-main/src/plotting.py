from typing import List
import numpy as np
from matplotlib import pyplot as plt
from contextlib import contextmanager
from scipy.signal import medfilt


def moving_average(X, window: int):
    cumsum_vec = np.cumsum(np.insert(X, 0, 0)) 
    return (cumsum_vec[window:] - cumsum_vec[:-window]) / window

def moving_median(X, window: int):
    return medfilt(X, kernel_size=window)


def plot_epsilons(epsilons):
    plt.figure()
    plt.title('Epsilons per episode')
    plt.xlabel('episode')
    plt.ylabel('epsilon')
    plt.plot(epsilons)


def plot_losses(losses):
    plt.figure()
    plt.title('Loss per update')
    plt.xlabel('update')
    plt.ylabel('loss')
    plt.plot(losses)
    plt.plot(moving_average(losses, 30))
    
    
def plot_gains(Gs):
    plt.figure()
    plt.title('Total gain per episode')
    plt.xlabel('episode')
    plt.ylabel('sum of gains')
    plt.plot(Gs)
    plt.plot(moving_median(Gs, 9))


def plot_distance_diffs(distance_diffs: List[float]):
    plt.figure()

    plt.subplot(311)
    plt.title('Total distance difference per episode')
    plt.xlabel('episode')
    plt.ylabel('distance difference (times)')
    plt.plot(distance_diffs)
    plt.plot(moving_median(distance_diffs, 9))

    plt.subplot(312)
    plt.title('Total distance difference per episode (zoomed)')
    plt.xlabel('episode')
    plt.ylabel('distance difference (times)')
    plt.plot(distance_diffs)
    plt.plot(moving_median(distance_diffs, 7))
    plt.ylim(-1, 2)

    plt.subplot(313)
    plt.title('Total distance difference per episode (zoomedx10)')
    plt.xlabel('episode')
    plt.ylabel('distance difference (times)')
    plt.plot(distance_diffs)
    plt.plot(moving_median(distance_diffs, 3))
    plt.ylim(-0.1, 0.2)


def plot_distance_diffs_relative(distance_diffs: List[float], n_nodes: List[int], type: str = 'training', max: int = None):
    plt.figure()
    plt.title(f'Total distance difference per episode ({type})')
    plt.xlabel('optimal path steps')
    plt.ylabel('distance difference (times)')
    plt.scatter(n_nodes, distance_diffs)

    if max:
        plt.ylim(-1, max)

    distance_diffs = np.array(distance_diffs)
    for n in np.unique(n_nodes):
        y = distance_diffs[np.argwhere(n_nodes == n)]
        plt.boxplot(y, positions=[n])


@contextmanager
def plot():
    try:
        yield None
    finally:
        plt.show()

import numpy
from abstract import Bandit


class NormalBandit(Bandit):
    def __init__(self, means, sigma, seed=None):
        '''Accept an array of K >= 2 floats, a float denoting the standard
        deviation and (optionally) a seed for a random number generator.'''
        self.means = means
        self.sigma = sigma
        self.random = numpy.random.RandomState(seed)
        
        self.k_star = numpy.argmax(means)
        self.gaps = means[self.k_star] - means
        self.regret = []
    
    def get_K(self) -> int:
        '''Return the number of actions.'''
        return len(self.means)

    def play(self, k: int, dry_run: bool = False) -> float:
        '''Accept a parameter 0 <= k < K, logs the instant pseudo-regret,
        and return the realization of a Normal random variable with mean
        of the given action and variance sigma.'''
        if not dry_run:
          self.regret.append(self.gaps[k])

        samples = self.random.normal(self.means, self.sigma, self.get_K())
        return samples[k]
    
    def get_cumulative_regret(self) -> numpy.ndarray:
        '''Return an array of the cumulative sum of pseudo-regret per round.'''
        return numpy.cumsum(self.regret)


class BernoulliBandit(Bandit):
    def __init__(self, means, seed=None):
        '''Accept an array of K >= 2 floats in [0, 1] and (optionally)
        a seed for a random number generator.'''
        self.means = means
        self.random = numpy.random.RandomState(seed)
        
        # for tracking regret
        self.k_star = numpy.argmax(means)
        self.gaps = means[self.k_star] - means
        self.regret = []
    
    def get_K(self) -> int:
        '''Return the number of actions.'''
        return len(self.means)

    def play(self, k: int, dry_run: bool = False) -> float:
        '''Accept a parameter 0 <= k < K, logs the instant pseudo-regret,
        and return the realization of a Bernoulli random variable with P(X=1)
        being the mean of the given action.'''
        if not dry_run:
          self.regret.append(self.gaps[k])

        samples = self.random.rand(self.get_K())
        reward = int(samples[k] < self.means[k])
        return reward
    
    def get_cumulative_regret(self) -> numpy.ndarray:
        '''Return an array of the cumulative sum of pseudo-regret per round.'''
        return numpy.cumsum(self.regret)

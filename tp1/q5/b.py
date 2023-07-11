import numpy
from os import path
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from .a import ContextualLinGaussianBandit, KernelTS


T = 1000
K = 10
sigma = 0.1

Rs = [sigma/2, sigma, sigma*2]

Lambda = sigma**2
thetas = numpy.random.uniform(-1, 1, (K, 5))
thetas = thetas / numpy.linalg.norm(thetas, 2, axis=1, keepdims=True)
seed = 43

for R in tqdm(Rs):
  bandit = ContextualLinGaussianBandit(thetas, sigma, seed=seed)

  models = [GaussianProcessRegressor(RBF(length_scale=1), alpha=Lambda, optimizer=None, random_state=seed) for k in range(K)]
  algo = KernelTS(Lambda=Lambda, R=R, S=1, models=models, seed=seed)
  algo.execute(bandit, T)

  cumul_regret = bandit.get_cumulative_regret()

  pyplot.plot(cumul_regret, label=algo.get_name())

pyplot.title(f'Comparaison des regrets selon les valeurs de R pour $\sigma$={sigma}')
pyplot.xlabel("Pas de temps")
pyplot.ylabel("Pseudo-regret cumulatif")
pyplot.legend()
pyplot.savefig(path.join(Path.cwd(), 'q5', 'b.png'))

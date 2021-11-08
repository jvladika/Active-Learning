from .abstract_sampler import AbstractSampler
import numpy as np


class RandomSampler(AbstractSampler):
    def __init__(self, X, y, seed=None, **kwargs):
        super().__init__(X, y, seed, **kwargs)
        self.name = 'random'

    def select_batch(self, labeled, N, **kwargs):
        all_inds = np.arange(self.X.shape[0])
        unlabeled = np.setdiff1d(all_inds, labeled)
        return np.random.choice(unlabeled, size=N, replace=False)
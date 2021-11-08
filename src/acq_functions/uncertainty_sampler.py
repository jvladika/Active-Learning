from .abstract_sampler import AbstractSampler
import numpy as np
import torch


class LeastConfidentSampler(AbstractSampler):
    def __init__(self, X, y, seed=None, **kwargs):
        super().__init__(X, y, seed, **kwargs)
        self.name = 'least_confident'

    def select_batch(self, model, labeled, N, **kwargs):
        all_inds = np.arange(self.X.shape[0])
        unlabeled = np.setdiff1d(all_inds, labeled)

        probs = model.predict_proba_numpy(self.X[unlabeled])
        max_probs = np.max(probs, axis=1)

        # Retrieve N instances with highest posterior probabilities.
        top_n = np.argpartition(max_probs, -N)[-N:]
        return unlabeled[top_n] 


class MarginSampler(AbstractSampler):
    def __init__(self, X, y, seed=None, **kwargs):
        super().__init__(X, y, seed, **kwargs)
        self.name = 'margin'

    def select_batch(self, model, labeled, N, **kwargs):
        all_inds = np.arange(self.X.shape[0])
        unlabeled = np.setdiff1d(all_inds, labeled)

        probs = model.predict_proba_numpy(self.X[unlabeled])

        sort_probs = np.sort(probs)[:, :2]
        min_margin = abs(sort_probs[:, 1] - sort_probs[:, 0])

        # Retrieve N instances with smallest margins.
        top_n = np.argpartition(min_margin, -N)[-N:]
        return unlabeled[top_n]


class EntropySampler(AbstractSampler):
    def __init__(self, X, y, seed=None, **kwargs):
        super().__init__(X, y, seed, **kwargs)
        self.name = 'entropy'

    def select_batch(self, model, labeled, N, **kwargs):
        all_inds = np.arange(self.X.shape[0])
        labeled = np.asarray(labeled)
        unlabeled = np.setdiff1d(all_inds, labeled)

        probs = model.predict_proba_numpy(self.X[unlabeled])
        
        # Clip for numerical stability.
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)

        # Retrieve N instances with highest entropies.
        top_n = np.argpartition(entropies, -N)[-N:]

        return unlabeled[top_n]

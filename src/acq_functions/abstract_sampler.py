from abc import (ABC, abstractmethod)

'''
Abstract class describing a general active-learning sampler.

Provides a function for selection of data points in a batch.
'''

class AbstractSampler(ABC):
    def __init__(self, X, y, seed, **kwargs):
        self.X = X
        self.y = y
        self.seed = seed
    
    @abstractmethod
    def select_batch(self):
        return
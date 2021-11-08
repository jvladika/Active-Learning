from .random_sampler import RandomSampler
from .uncertainty_sampler import (LeastConfidentSampler, MarginSampler, EntropySampler)
from .query_by_committee import QBC
from .bald import Bald


AL_MAPPING = {
    'random': RandomSampler,
    'least_confident': LeastConfidentSampler,
    'margin': MarginSampler,
    'entropy': EntropySampler,
    'qbc': QBC,
    'bald': Bald
}

def get_AL_sampler(name):
  if name in AL_MAPPING:
    return AL_MAPPING[name]
  
  raise NotImplementedError(f'The specified sampler "{name}" is not available.')
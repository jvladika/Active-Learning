# Run mutliple experiments with different parameters
# define all parameters for grid search here

# In case of checkpoint-restart, provide the path to log file as argument

from generic_experiment import AlExperiment
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import constants
import al_config
import threading
import os
import sys
import psutil
import itertools
import utils
import csv
import numpy as np

import al_config

L = utils.logger('experiment_runner')
columns = ['model', 'al_method', 'warmstart_size', 'batch_size', 'runs']

AL_METHOD = al_config.samplers
WARMSTART_SIZE = al_config.warmstart_size
BATCH_SIZE = al_config.batch_size
GRID_SEARCH_STEP = al_config.grid_search_step
SEEDS = al_config.seeds
MODEL = al_config.models
SAVE_SEEDS = al_config.save_seeds

grid_search = list(itertools.product(*[MODEL, AL_METHOD, WARMSTART_SIZE, BATCH_SIZE]))
CPU_COUNT = al_config.workers


class Runner:

    def __init__(self, data, dataset_name, multilabel=False,
                 checkpoint_file=None):
        self.data = data
        self.dataset_name = dataset_name
        self.multilabel = multilabel
        
        self.checkpoint_file = checkpoint_file
        self.completed = set()
        if self.checkpoint_file is not None:
            # load checkpoint
            with open(os.path.join(os.path.dirname(__file__), checkpoint_file), 'r') as f:
                reader = csv.reader(f)
                # skip header
                next(reader, None)
                for row in reader:
                    self.completed.add((row[0], row[1], int(row[2]), int(row[3]), int(row[4])))            
            L.info('Loaded checkpoint, %d experiments completed.' % (len(self.completed)))
        else:
            self.checkpoint_file = 'checkpoint.csv'
            self.write_header(self.checkpoint_file)
            L.info('Checkpoint file created: %s.' % checkpoint_file)


    def run(self, verbose=True):

        with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
            for model, al_method, warmstart_size, batch_size in grid_search:

                if (model, al_method, warmstart_size, batch_size, len(SEEDS)) in self.completed:
                    continue
                
                expr = AlExperiment(data=self.data,
                                    dataset_name=self.dataset_name,
                                    al_method=al_method,
                                    warmstart_size=warmstart_size,
                                    batch_size=batch_size,
                                    grid_search_step=GRID_SEARCH_STEP,
                                    seeds=SEEDS,
                                    multilabel=self.multilabel,
                                    model_name=model,
                                    save_seeds=SAVE_SEEDS)
                executor.submit(expr.run, checkpoint_file, verbose=verbose)
                #expr.run(checkpoint_file=self.checkpoint_file, verbose=verbose)

    @staticmethod
    def write_header(checkpoint_file):
        L.debug("Writing header")
        with open(checkpoint_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(columns)



if __name__ == '__main__':
    # Limit the number of used CPU's -> assign CPU affinity

    if al_config.cpus is not None:
        p = psutil.Process(os.getpid())
        p.cpu_affinity(al_config.cpus)

    if len(sys.argv) == 2:
        checkpoint_file = sys.argv[1]
    else:
        checkpoint_file = al_config.checkpoint

    path_pkl = al_config.datapath
    dataset_name = al_config.dataset_name
    data = utils.load(path_pkl, mode='pickle')

    Runner(data, dataset_name, multilabel=True, checkpoint_file=checkpoint_file).run(verbose=False)


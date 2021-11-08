import argparse
import csv
import datetime
import math
import os
import time
import warnings
from functools import partial

import numpy as np
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split, PredefinedSplit
from sklearn.utils import shuffle

import json

import torch
from torch._C import dtype
import torch.nn as nn
import torch.optim as optim

from podium.datasets import SingleBatchIterator
from podium.vectorizers import GloVe

import utils
from metrics import Metrics
from stopping_criteria import QueryCount, VarianceSurge
from acq_models.models import TorchModel, get_model
from acq_functions.samplers import get_AL_sampler
from acq_functions.random_sampler import RandomSampler
from acq_functions.uncertainty_sampler import LeastConfidentSampler

from al_config import STORAGE_PATH
from data import (TorchTM, SklearnTM)

warnings.warn = lambda *a, **kw: None

L = utils.logger('al_experiment')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', type=str, default='imdb',
                    help='Dataset name.')
parser.add_argument('--max-vocab-size', type=int, default=25_000,
                    help='Maximum vocab size.')
parser.add_argument('--acq-model', type=str, default='log_reg',
                    help='AL acqusition model.')
parser.add_argument('--num-epoch', type=int, default=20,
                    help='Number of epoch to train the model in each AL step.')
parser.add_argument('--embedding-dim', type=int, default=100,
                    help='Embedding dimensionality.')
parser.add_argument('--acq-fun', type=str, default='random',
                    help='AL acquisition function.')
parser.add_argument('--al-batch-size', type=int, default=10_000,
                    help='AL batch size.')
parser.add_argument('--warmstart-size', default=20,
                    help=("Can be float or integer. Float indicates percentage of training data "
						  "to use in the initial warmstart model. Use -1 for cold start (one example from each class)."))

# parser.add_argument('--grid-search-step', type=int, default=-1,
#                     help='Perform grid search CV for optimal parameters every n batches. Use -1 to disable.')
# parser.add_argument('--multilabel', type=bool, default=False,
#                   help='True for multilabel data, false otherwise.')
# parser.add_argument('--seed', type=int, default=42,
#                     help='Seed.')

parser.add_argument('--log-every', type=int, default=10,
                    help='Results are logged periodically.')


RESULT_PATH = STORAGE_PATH

'''
Main class that contains functions and variables needed for runnning 
the Active Learning experiment.
'''

class AlExperiment():

    def __init__(self,
                 train_manager,            #object that wraps the training parameters and settings
                 dataset_name,             #dataset to test on
                 al_method,                #active learning sampling method used ('entropy', 'margin', 'qbc', 'bald')
                 warmstart_size,           #number of data points used for pretraining the first model
                 al_batch_size,            #number of data points given to annotators in each step
                 grid_search_step=-1,      #step used when optimizing the parameters using grid search
                 seeds=[42],               #random seeds used in each run, to get stochastic results
                 model_name='log_reg',     #ML model used in the experiment ('log_reg', 'svm', 'rnn')
                 model=None,               #ML model object
                 save_seeds=False):

        self.train_manager = train_manager             
        self.dataset_name = dataset_name
        self.al_method = al_method
        self.warmstart_size = warmstart_size
        self.al_batch_size = al_batch_size
        self.seeds = seeds
        self.grid_search_step = grid_search_step
        self.model_name = model_name
        self.model = model
        self.info = None
        self.results = []
        self.save_seeds = save_seeds
        self.ml_metrics = Metrics()

    '''
    Runs one iteration of an Active Learning experiment. 
    '''
    def run_single(self, seed=42, verbose=False, logiter=5):
        utils.set_seed_everywhere(seed)

        # Torch neural models needs to reset neurons' weights in each step in order to re-train.
        if isinstance(self.model, TorchModel):
            for layer in self.model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        # Load the train and test data.
        X_train, y_train = self.train_manager.get_numpy_data('train')
        X_test, y_test = self.train_manager.get_numpy_data('test')
        train_size = self.train_manager.dataset_size('train')
        test_size = self.train_manager.dataset_size('test')
        
        # Get the sampler for this particular active-learning samplin strategy.
        sampler = get_AL_sampler(self.al_method)
        al_batch_size = self.check_percentage(self.al_batch_size, train_size)
        warmstart_size = self.check_percentage(self.warmstart_size, train_size)
        stopping_criterion = QueryCount

        # Load the model.
        model = self.model

        if verbose:
            L.debug(f'train_size: {train_size}, test_size: {test_size}')
            L.debug(f'al_method:{self.al_method}  warmstart_size:{warmstart_size} '
                    f'al_batch_size:{al_batch_size} seed:{seed} model:{self.model_name}')

        results = {}
        metrics_train = []
        metrics_test = []
        predictions_test = []

        # TODO: implement stratified sample for torch
        selected_inds = np.random.choice(np.arange(train_size), warmstart_size, replace=False).tolist()

        # initial sample
        # if warmstart_size == -1:
        #     # cold start - use one example from each class
        #     selected_inds = utils.stratified_sample(
        #         y_train, np.unique(y_train).size)
        # else:
        #     # use warmstart_size/class_count examples from each class
        #     selected_inds = utils.stratified_sample(y_train, warmstart_size)

        seed_batch = len(selected_inds)
        num_labeled = [seed_batch]
        stopping_criterion = stopping_criterion(train_size=train_size,
                                                batch_size=al_batch_size,
                                                seed_batch=seed_batch)
                                                #, X=X_train, y=y_train)
        sampler = sampler(X_train, y_train, seed)

        been = True
        while not stopping_criterion.is_over():
            curr_batch = stopping_criterion.current_batch
            n_train = seed_batch + \
                min(train_size - seed_batch, curr_batch * al_batch_size)

            if verbose:
                L.debug('Training model on %s/%s datapoints',
                        n_train, train_size)

            assert n_train == len(selected_inds)

            # Sort active_ind so that the end results matches that of uniform sampling.
            sorted(selected_inds)
            X_selected_batch = X_train[selected_inds]
            y_selected_batch = y_train[selected_inds]

            # Train the model with selected data points!
            model.al_step(selected_inds, self.train_manager)

            # TODO: transfer to al step
            metrics_train.append(
                self.ml_metrics.eval_metrics(
                    ys_true=y_selected_batch, hs=model.predict_numpy(X_selected_batch)))

            metrics_test.append(
                self.ml_metrics.eval_metrics(
                    ys_true=y_test, hs=model.predict_numpy(X_test)))

            if (curr_batch + 1) % logiter == 0 and verbose:
                L.debug(f'Sampler: {sampler.name}, metrics_train: {metrics_train[-1]}, metrics_test: {metrics_test[-1]}')


            # The data in next steps will be selected from the remainder of unlabeled data points.
            remaining = train_size - len(selected_inds)
            n_sample = min(al_batch_size, remaining)
            if n_sample == remaining:
                all_inds = np.arange(train_size)
                new_batch = np.setdiff1d(all_inds, selected_inds).tolist()
            elif n_sample > 0:
                # These inputs will be provided to various different samplers,
                # depending on their respective method signatures.
                select_args = dict(model=model,
                                sampler=sampler,
                                N=n_sample,
                                labeled=selected_inds,
                                X_test=X_test,
                                y_test=y_test,
                                verbose=verbose)
                #model = model.train()
                new_batch = sampler.select_batch(**select_args).tolist()
            else:
                new_batch = []

            selected_inds.extend(new_batch)
            num_labeled.append(len(selected_inds))

            if verbose:
                L.info('Requested: %d, Selected: %d' %
                        (n_sample, len(new_batch)))
            assert len(new_batch) == n_sample
            assert len(set(selected_inds)) == len(selected_inds)

            stopping_criterion.next_state(annotated_count=n_sample)

        # After the stopping criterion is reached, the active learning process stops and results are ready.
        stopping_criterion.reset()
        results = [{'metrics_train': m_train, 'metrics_test': m_test, 'labeled': labeled}
                   for m_train, m_test, labeled in zip(metrics_train, metrics_test, num_labeled)]

        # Make results readable and save them.
        result = dict(meta=dict(dataset=self.dataset_name,
                                sampler=self.al_method,
                                warmstart_size=warmstart_size,
                                batch_size=al_batch_size,
                                model=self.model_name,
                                grid=self.grid_search_step,
                                seed=seed),
                      results=results)


        if self.save_seeds:
            filepath = RESULT_PATH + \
                f'{self.dataset_name}.{self.al_method}.{self.model_name}.ws={warmstart_size}.bs={al_batch_size}.s={seed}.grid={self.grid_search_step}.{utils.time_string()}.json'
            utils.dump(result, filepath, overwrite=True, mode='json')

            if verbose:
                L.info('Results stored to %s', filepath)

        return result

    '''
    Run multiple AL experiments, once for each seed. 
    '''
    def run(self, checkpoint_file=None, verbose=False, logiter=5):
        for seed in self.seeds:
            L.debug('Running experiment : %s' % self.info_string(seed))
            self.start_time()
            result = self.run_single(seed, verbose, logiter)
            self.results.append(result)
            self.end_time()
            L.debug('Finished experiment : %s in %s' %
                    (self.info_string(seed), self.get_duration()))

        if checkpoint_file is not None:
            self.write_to_csv(checkpoint_file)
        self.store_average()


    def check_percentage(self, sample, total_len):
        if sample < 1 and sample != -1:
            return int(sample*total_len)
        else:
            return int(sample)

    '''
    Store the average results of multiple runs (with multiple seeds),
    '''
    def store_average(self, metrics=[(lambda r: r['metrics_test'].aggregated['f1_micro'], 'f1_micro'),
                                     (lambda r: r['metrics_test'].aggregated['f1_macro'], 'f1_macro')]):
        seeds = []
        xs = [r['labeled'] for r in self.results[0]['results']]
        Ys = []
        for result in self.results:
            seeds.append(result['meta']['seed'])
            res = result['results']
            Y_met = [[] for _ in range(len(metrics))]
            for r in res:
                for j, metric in enumerate(metrics):
                    val = metric[0](r)
                    Y_met[j].append(val)
            Ys.append(Y_met)

        Ys = np.array(Ys).swapaxes(0, 1)

        results = dict(labeled=xs)
        for i, metric in enumerate(metrics):
            results[metric[1]] = Ys[i].mean(0)
            results[f'{metric[1]}_var'] = Ys[i].var(0)
            results[f'{metric[1]}_baseline'] = Ys[i][0][-1]

        result_avg = dict(meta=dict(dataset=self.dataset_name,
                                    sampler=self.al_method,
                                    warmstart_size=self.warmstart_size,
                                    batch_size=self.al_batch_size,
                                    model=self.model_name,
                                    grid=self.grid_search_step,
                                    seeds=seeds),
                          results=results)


        filepath = os.path.join(
                        RESULT_PATH,
                        f'avg.{self.dataset_name}.{self.al_method}'
                        f'.ws={self.warmstart_size}'
                        f'.bs={self.al_batch_size}'
                        f'.grid={self.grid_search_step}'
                        f'.{utils.time_string()}.json')
        utils.dump(result_avg, filepath, overwrite=True, mode='json')
        L.info('Average stored to %s', os.path.abspath(filepath))

    def info_string(self, seed):
        self.info = f'(%s, %d, %d, %s, %d)' % \
            (self.al_method, self.warmstart_size,
             self.al_batch_size, self.model_name, seed)
        return self.info

    def start_time(self):
        self.start = time.time()

    def end_time(self):
        self.end = time.time()
        self.duration = str(datetime.timedelta(
            seconds=math.floor(self.end-self.start)))

    def get_duration(self):
        try:
            return self.duration
        except:
            raise Exception(
                'start_time and end_time must be called respectively!')

    def write_to_csv(self, file):
        with open(file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow((self.model_name, self.al_method,
                             self.warmstart_size, self.al_batch_size, len(self.seeds)))


'''
Perform the active learning experiment.

Receives train-validation-test data splits, dataset, and word vocabulary.
'''
def perform(splits, dataset, vocab):
     #splits, fields, vocab = utils.load_trec()
    splits[0].finalize_fields()
    
    SEED = 42
    utils.set_seed_everywhere(SEED)

    sets = tuple([split.batch() for split in splits])
    vectorizer = GloVe(dim=args.embedding_dim)

    """
    def device_tensor(data):
        return torch.tensor(data, dtype=torch.float).to(torch.device('cpu'))
    
    # Torch example.
    # ===========================================================
    model_name = 'rnn'
    padding_idx = vocab.get_padding_index()

    device = torch.device('cpu')
    emb_matrix = torch.tensor(vectorizer.load_vocab(vocab),
                              dtype=torch.float,
                              device=device)

    model_params = dict(
        embedding_dim=args.embedding_dim,
        hidden_dim=50,
        output_dim=6,
        pretrained_embeddings=emb_matrix,
        padding_idx=padding_idx,
        device=device
    )

    criterion = nn.CrossEntropyLoss()
    model = get_model(name=model_name, params=model_params, scheme=None)
    model = model.to('cpu')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    manager = TorchTM(splits, criterion, optimizer, args.num_epoch)
    # ===========================================================
    
    
    """
    # Sklearn example.
    # ===========================================================
    model_name = 'svm'
    model_params = {"probability" : True, "C": 100, "gamma": 0.001} #"probability" : True, "C": 100, "gamma": 0.001
    model = get_model(name=model_name, params=model_params, scheme=None)
    emb_matrix = np.array(vectorizer.load_vocab(vocab))
    manager = SklearnTM(splits, emb_matrix)
    # ===========================================================
    

    AlExperiment(train_manager=manager,
                 dataset_name=dataset,
                 al_method='entropy',
                 warmstart_size=50,
                 al_batch_size=50,
                 grid_search_step=-7,
                 seeds=list(range(5)),
                 model_name=model_name,
                 model=model).run(verbose=True)


if __name__ == '__main__':
    args = parser.parse_args()

    dataset = 'sst'
    splits, fields, vocab = utils.load_data(dataset, field_names=["label", "text"], validation=True)
    
    perform(splits, dataset, vocab)
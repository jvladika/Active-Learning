from abc import (ABC, abstractmethod)
from podium.datasets.dataset import Dataset
from podium.datasets.iterator import BucketIterator, Iterator
import numpy as np
import torch


'''
Abstract class that is later inherited by SkLearn and Torch training manager.

Contains functions for accessing data of a particular split (train, validation, test) 
in Pythonic format and NumPy format.
'''
class TrainManager(ABC):
    @abstractmethod
    def __init__(self, sets, *args, **kwargs):
        if len(sets) == 3:
            self.train, self.valid, self.test = sets
        else:
            self.train, self.test = sets
        
        self.train_size = len(self.train)
        self.test_size = len(self.test)

    @abstractmethod
    def get_data(self, mode):
        pass

    @abstractmethod
    def get_partial_data(self, indices, mode):
        pass

    @abstractmethod
    def get_numpy_data(self, mode):
        pass

    def dataset_size(self, mode='train'):
        return len(getattr(self, mode))


class SklearnTM(TrainManager):
    def __init__(self, sets, emb_matrix):
        super().__init__(sets)
        data_train = self.train.batch(add_padding=False)
        data_test = self.test.batch(add_padding=False)

        self.X_train, self.y_train = data_train.get('sentence1'), data_train.get('label')
        self.X_test, self.y_test = data_test.get('sentence1'), data_test.get('label')

        """
        first, second, self.y_train = data_train.get('sentence1'), data_train.get('sentence2'), data_train.get('label')
        self.X_train = list()
        for a, b in zip(first, second):
            self.X_train.append(np.concatenate([a, b]))

        third, fourth, self.y_test = data_test.get('sentence1'), data_test.get('sentence2'), data_test.get('label')
        self.X_test = list()
        for a, b in zip(third, fourth):
            self.X_test.append(np.concatenate([a, b]))
        """
        
        try:
            self.X_train = emb_matrix[self.X_train].sum(axis=1)
            self.X_test = emb_matrix[self.X_test].sum(axis=1)
        except:
            train_result = np.reshape(emb_matrix[self.X_train[0]].sum(axis=0), (1, 100))
            for row in self.X_train[1:]:
                row_embedding = np.reshape(emb_matrix[row].sum(axis=0), (1,100))
                train_result = np.concatenate((train_result, row_embedding))
            
            test_result = np.reshape(emb_matrix[self.X_test[0]].sum(axis=0), (1, 100))
            for row in self.X_test[1:]:
                row_embedding = np.reshape(emb_matrix[row].sum(axis=0), (1,100))
                test_result = np.concatenate((test_result, row_embedding))
            
            self.X_train = train_result
            self.X_test = test_result
        
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

    def get_numpy_data(self, mode):
        return self.get_data(mode)

    def get_data(self, mode):
        X = getattr(self, f'X_{mode}')
        y = getattr(self, f'y_{mode}')
        return X, y

    def get_partial_data(self, indices, mode='train'):
        indices = np.array(indices, dtype=np.int)
        X, y = self.get_data(mode)
        return X[indices], y[indices]


class TorchTM(TrainManager):
    def __init__(self, sets, optimizer, criterion, num_epochs=1, batch_size=64, device=torch.device('cpu')):
        super().__init__(sets)
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

    def device_tensor(self, data):
        return torch.tensor(data, dtype=torch.long).to(self.device)

    def get_data(self, indices, mode='train'):
        data = getattr(self, mode)
        return Iterator(
                data,
                batch_size=self.batch_size,
                matrix_class=self.device_tensor
               )

    def get_partial_data(self, indices, mode='train'):
        set_ = getattr(self, mode)
        examples = [set_[i] for i in indices]
        data = Dataset(examples, fields=set_.fields)
        return Iterator(
                data,
                batch_size=self.batch_size,
                matrix_class=self.device_tensor,
               )

    def get_numpy_data(self, mode):
        train = getattr(self, f'{mode}')
        batch = train.batch(add_padding=True)
        return batch.__getattr__(sorted(batch.keys())[1]), \
                    batch.__getattr__(sorted(batch.keys())[0])

    @staticmethod
    def text_len_sort_key(example):
        tokens = example['text'][1]
        return -len(tokens)

    @staticmethod
    def extract_fields(dataset):
        return dataset.text, dataset.label

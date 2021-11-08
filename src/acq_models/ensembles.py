import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import abc
from functools import partial

'''
Abstract class for ensembles, used for the Query-by-Committee active learning sampling strategy.
'''
class EnsembleBase(abc.ABC, nn.Module):

    def __init__(self, estimator, n_estimators, output_dim,
                 epochs=1, n_jobs=1, optimizer=optim.Adam,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.output_dim = output_dim
        
        self.epochs = epochs
        
        self.n_jobs = n_jobs
        self.device = device
        
        # Initialize base estimators.
        mod_list = [estimator().to(self.device) for _ in range(self.n_estimators)]
        self.estimators_ = nn.ModuleList(mod_list)
        
        self.optimizer = optimizer(self.parameters())
    
    def __str__(self):
        info = []
        info.append(50*'=')
        info.append(f'{"Base Estimator":<20}: {self.estimator.__name__}')
        info.append(f'{"n_estimator":<20}: {self.n_estimators}')
        s = '\n'.join(info)
        return s
    
    def __repr__(self):
        return self.__str__()
    
    def _validate_parameters(self):
        # TODO: implement parameter validation
        pass
    
    @abc.abstractmethod
    def forward(self, X):
        pass
    
    @abc.abstractmethod
    def fit(self, train_loader):
        pass
    
    @abc.abstractmethod
    def predict(self, test_loader):
        pass


class BaggingClassifier(EnsembleBase):
    
    def forward(self, X):
        batch_size = X.size()[0]
        y_pred = torch.zeros(batch_size, self.output_dim).to(self.device)
        
        # Average over the class distributions predicted from all of the base estimators.
        for estimator in self.estimators_:
            y_pred += estimator(X)
        y_pred /= self.n_estimators
        
        return y_pred
    
    def fit(self, train_loader):
        
        self.train()
        self._validate_parameters()
        criterion = nn.CrossEntropyLoss()
        
        # TODO: Parallelization
        for est_idx, estimator in enumerate(self.estimators_):
            
            # Initialize an independent optimizer for each base estimator to 
            # avoid unexpected dependencies.
            estimator_optimizer = torch.optim.Adam(estimator.parameters(),
                                                   lr=self.lr,
                                                   weight_decay=self.weight_decay)
        
            for epoch in range(self.epochs):
                for batch_idx, (X_train, y_train) in enumerate(train_loader):
                    
                    batch_size = X_train.size()[0]
                    X_train, y_train = (X_train.to(self.device), 
                                        y_train.to(self.device))
                    
                    loss = torch.tensor(0.).to(self.device)
                    
                    # In `BaggingClassifier`, each base estimator is fitted on a 
                    # batch of data after sampling with replacement.
                    sampling_mask = torch.randint(high=batch_size, 
                                                  size=(int(batch_size),), 
                                                  dtype=torch.int64)
                    # sampling_mask = torch.unique(sampling_mask)  # remove duplicates
                    sampling_X_train = X_train[sampling_mask]
                    sampling_y_train = y_train[sampling_mask]
                    
                    sampling_output = estimator(sampling_X_train)
                    loss += criterion(sampling_output, sampling_y_train)
                        
                    estimator_optimizer.zero_grad()
                    loss.backward()
                    estimator_optimizer.step()
                    
                    # Print training status
                    # if batch_idx % self.log_interval == 0:
                    #     y_pred = F.softmax(sampling_output, dim=1).data.max(1)[1]
                    #     correct = y_pred.eq(sampling_y_train.view(-1).data).sum()
                        
                    #     msg = ('Estimator: {:03d} | Epoch: {:03d} |' 
                    #            ' Batch: {:03d} | Loss: {:.5f} | Correct:'
                    #            ' {:d}/{:d}')
                    #     print(msg.format(est_idx, epoch, batch_idx, loss, 
                    #                      correct, y_pred.size()[0]))
    
    def predict(self, test_loader):
        
        self.eval()
        correct = 0.

        for batch_idx, (X_test, y_test) in enumerate(test_loader):
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            output = self.forward(X_test)
            y_pred = output.data.max(1)[1]
            correct += y_pred.eq(y_test.view(-1).data).sum()
        
        accuracy = 100. * float(correct) / len(test_loader.dataset)

        return accuracy
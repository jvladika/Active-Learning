from abc import (ABC, abstractmethod)
from functools import partial

import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier as OVO
from sklearn.multiclass import OneVsRestClassifier as OVR


RNN_TYPES = ['RNN', 'LSTM', 'GRU']


class AbstractModel(ABC):
    @abstractmethod
    def predict_proba(self, batch):
        pass

    @abstractmethod
    def predict(self, batch):
        pass

    @abstractmethod
    def al_step(self, X, y, **kwargs):
        pass


class ScikitModel(ABC):
    def al_step(self, indices, train_manager):
        X, y = train_manager.get_partial_data(indices)
        self.fit(X, y.reshape(-1, 1))

    def predict_numpy(self, X):
        return self.predict(X)

    def predict_proba_numpy(self, X):
        return self.predict_proba(X)


class TorchModel(ABC):
    def predict_proba(self, X):
        self.eval()
        y_pred = self.forward(X)
        if self.output_dim == 1:
            #y_pred = torch.sigmoid(y_pred)
            y_pred = torch.cat([1.-y_pred, y_pred], dim=1)
        else:
            y_pred = F.softmax(y_pred, dim=1)
        self.train()
        return y_pred

    def predict(self, X):
        self.eval()
        
        """
        dataloader = torch.utils.data.DataLoader(X,
            batch_size=128, pin_memory=True, shuffle=True)

        prediction_list = []
        for i, batch in enumerate(dataloader):
            pred = self.forward(batch)
            prediction_list.extend(pred.cpu())

        y_pred = torch.stack(prediction_list)
        """

        y_pred = self.forward(X)
        if self.output_dim == 1:
            #y_pred = torch.sigmoid(y_pred)
            out = torch.as_tensor((y_pred - 0.5) > 0,
                                  dtype=torch.long, device=self.device)
        else:
            #y_pred = F.softmax(y_pred, dim=1)
            out = torch.argmax(y_pred, dim=1)
        
        self.train()
        return out

    def predict_proba_numpy(self, X):
        X = torch.tensor(X)
        with torch.no_grad():
            y_pred = self.predict_proba(X)
            return y_pred.numpy()

    def predict_numpy(self, X):
        X = torch.tensor(X)
        with torch.no_grad():
            out = self.predict(X)
            return out.numpy()
    
    def evaluate(model, iterator, criterion):
        # TODO: add F1 metric
        num_batches = len(iterator)
        model.eval()
        
        with torch.no_grad():
        
            for batch_index, batch in enumerate(iterator, 1):
                y_pred = model(batch)

                loss = criterion(y_pred.squeeze(), batch.label)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / batch_index
                
                acc_t = accuracy(y_pred, batch.label)
                running_acc += (acc_t - running_acc) / batch_index
        
        model.train()
        return running_loss, running_acc

    def train_loop(self, iterator, optimizer, criterion, num_epochs):
        for _ in range(num_epochs):
            for batch in iterator:
                X, y = batch.__getattr__(sorted(batch.keys())[1]), \
                       batch.__getattr__(sorted(batch.keys())[0])
                # 5 step training routine
                # ------------------------------------------

                # 1) zero the gradients
                optimizer.zero_grad()
                
                # 2) compute the output
                y_pred = self(X)
                
                # 3) compute the loss
                loss = criterion(y_pred.squeeze(), y.squeeze())
                
                # 4) use loss to produce gradients
                loss.backward()

                # 5) use optimizer to take gradient step
                optimizer.step()
            
    def al_step(self, indices, train_manager):
        iterator = train_manager.get_partial_data(indices)
        self.train()
        self.train_loop(iterator,
                        train_manager.criterion,
                        train_manager.optimizer,
                        train_manager.num_epochs)
        


class MLP(nn.Module, TorchModel):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 pretrained_embeddings, padding_idx=1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.output_dim = output_dim
        self.device = device

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings,
                                                      padding_idx=padding_idx)

        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        embedded = self.embedding(X).sum(axis=1)
        l1 = F.relu(self.hidden(embedded))
        l2 = F.relu(self.hidden2(l1))
        out = self.out(l2)
        return out


class RNN(nn.Module, TorchModel):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 pretrained_embeddings, num_layers=1, bidirectional=True,
                 dropout_p=0.5, padding_idx=1, rnn_type='LSTM',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        super().__init__()
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.device = device

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings,
                                                      padding_idx=padding_idx)

        drop_prob = 0. if num_layers > 1 else dropout_p
        assert rnn_type in RNN_TYPES, f'Use one of the following: {str(RNN_TYPES)}'
        RnnCell = getattr(nn, rnn_type)
        self.rnn = RnnCell(embedding_dim,
                           hidden_dim,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=drop_prob,
                           batch_first=True)

        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, X):
        # X: B x S
        # print(f'X {X.shape}')

        #X = X.cuda()

        # embedded: B x S x E
        embedded = self.dropout(self.embedding(X))
        embedded = self.embedding(X)
        # print(f'embedded {embedded.shape}')

        # out: B x S x (H*num_directions)
        # hidden: B x (L*num_directions) x H
        out, hidden = self.rnn(embedded)
        # print(f'hidden {hidden.shape}')
        if type(hidden) == tuple: hidden = hidden[0]

        # if bidirectional concat the final forward (hidden[-1]) and
        # backward (hidden[-2]) hidden layers, otherwise extract the
        # final hidden state and apply dropout
        # hidden = B x (H*num_directions)
        if self.bidirectional:
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=-1)
        else:
            hidden = hidden[:,-1,:]

        hidden = self.dropout(hidden)
        # print(f'hidden {hidden.shape}')

        out = self.fc(hidden)
        # print(f'out {out.shape}')
        return F.softmax(out)


class PackedRNN(RNN):
    def forward(self, batch):
        # X: S x B
        X, lengths = batch

        # embedded: S x B x E
        embedded = self.dropout(self.embedding(X))

        # pack sequence
        # output over padding tokens are zero tensors
        # hidden: (L*num_directions) x B x H
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        packed_out, hidden = self.rnn(packed_embedded)
        if type(hidden) == tuple: hidden = hidden[0]

        # unpack sequence
        # out: S x B x (H*num_directions)
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_out)

        # if bidirectional concat the final forward (hidden[-2,:,:]) and
        # backward (hidden[-1,:,:]) hidden layers, otherwise extract the
        # final hidden state and apply dropout
        # hidden = B x (H*num_directions)
        if self.bidirectional:
            hidden = torch.cat((hidden[:, -2, :], hidden[:, -1, :]), dim=1)
        else:
            hidden = hidden[:, -1, :]

        hidden = self.dropout(hidden)
        return self.fc(hidden)


class BertClassifier(nn.Module, TorchModel):
    def __init__(self, output_dim, bert_model_name='bert-base-uncased'):
        super().__init__()
        self.output_dim = output_dim
        self.bert = BertModel.from_pretrained(bert_model_name)
        config = self.bert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, output_dim)
        # Xavier initalization
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, X):
        out = self.bert(X)
        print(out)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


class LogReg(ScikitModel, LogisticRegression):
    pass


class SVM(ScikitModel, SVC):
    pass

class RandomForest(ScikitModel, RandomForestClassifier):
    pass


MODELS = {
    'log_reg': LogReg,
    'svm': SVM,
    'rnn': RNN,
    'rf' : RandomForest
}

#parametri su dict pa se za svaki model moze poslati
#kod grid searcha isto slati parametre
def get_model(name, params=dict(), scheme=None):

    if name not in MODELS:
        raise ValueError(f'Model "{name}" not supported')

    model = MODELS[name](**params)

    if scheme is None:
        return model
    elif scheme == 'ovr':
        return OVR(model)
    elif scheme == 'ovo':
        return OVO(model)
    else:
        raise ValueError(f'Invalid scheme "{scheme}"')
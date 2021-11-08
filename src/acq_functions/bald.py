import sys
sys.path.append(".")

from .abstract_sampler import AbstractSampler

from scipy.stats import norm
import numpy as np
import torch

def H(x, eps=1e-6):
    """ Compute the element-wise entropy of x
    Arguments:
        x {torch.Tensor} -- array of probabilities in (0,1)
    Keyword Arguments:
        eps {float} -- prevent failure on x == 0
    Returns:
        torch.Tensor -- H(x)
    """
    return -(x+eps)*torch.log(x+eps)


'''
Implementation of a Bayesian approach to active learning as described in 

"BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning" (2020)
https://arxiv.org/abs/1906.08158

'''

class Bald(AbstractSampler):

    def __init__(self, X=None, y=None, seed=None, batch_size=50):
        super(Bald, self).__init__(X, y, seed)
        self.batch_size = batch_size
        self.processing_batch_size = 1024
        self.device='cpu'
        self.name = 'bald'

    def select_batch(self, model, labeled, N, n_jobs=None, **kwargs):
        all_inds = np.arange(self.X.shape[0])
        labeled = np.asarray(labeled)
        unlabeled = np.setdiff1d(all_inds, labeled)

        X_unlab = self.X[unlabeled]
        y_unlab = self.y[unlabeled]
        #####################################

        pool_data = torch.utils.data.TensorDataset(torch.from_numpy(X_unlab), torch.from_numpy(y_unlab))

        pool_loader = torch.utils.data.DataLoader(pool_data,
            batch_size=self.processing_batch_size, pin_memory=True, shuffle=False)
        
        
        scores = torch.zeros(len(pool_data)).to(self.device)
        print("started scores")
        for batch_idx, (data, _) in enumerate(pool_loader):
            end_idx = batch_idx + data.shape[0]
            scores[batch_idx:end_idx] = self.score(model, data.to(self.device))

        best_local_indices = torch.argsort(scores)[-self.batch_size:]
        return unlabeled[best_local_indices]


    def score(self, model, x, k=50):
         # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]

        with torch.no_grad():
            # take k monte-carlo samples of forward pass w/ dropout

            dataloader = torch.utils.data.DataLoader(x,
            batch_size=128, pin_memory=False, shuffle=True)

            prediction_list = []
            for i, batch in enumerate(dataloader):
                pred = torch.stack([model(batch) for i in range(k)], dim=1)
                prediction_list.extend(pred)

            Y = torch.stack(prediction_list)
            #Y = torch.stack([model(x) for i in range(k)], dim=1)
            H1 = H(Y.mean(axis=1)).sum(axis=1)
            H2 = H(Y).sum(axis=(1,2))/k

            return H1 - H2

        '''
        C = np.sqrt((np.pi*np.log(2)) / 2.0)
        p = norm.cdf(np.sqrt(mean / (variance+1)))
        first = h(p)
        
        nom = C * np.exp(-mean**2 / (2*(variance**2 + C**2)))
        denom = np.sqrt(variance**2 + C**2)
        second = nom/denom

        return first-second
        '''
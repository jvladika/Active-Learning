from torch.nn.modules.loss import BCELoss
from .abstract_sampler import AbstractSampler
from sklearn.ensemble import BaggingClassifier as SkBagging
from torchensemble import BaggingClassifier as TorchBagging   
from acq_models.models import ScikitModel, TorchModel
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import collections
import torch
import time

class QBC(AbstractSampler):
    """The Query-By-Committee (QBC) algorithm.

    QBC minimizes the version space, which is the set of hypotheses that are consistent
    with the current labeled training data.

    This class implements the query-by-bagging method. Which uses the bagging in sklearn to
    construct the committee. So your model should be a sklearn model.

    Parameters
    ----------
    X: 2D array
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    method: str, optional (default=query_by_bagging)
        Method name. This class only implement query_by_bagging for now.

    disagreement: str
        method to calculate disagreement of committees. should be one of ['vote_entropy', 'KL_divergence']

    References
    ----------
    [1] H.S. Seung, M. Opper, and H. Sompolinsky. Query by committee.
        In Proceedings of the ACM Workshop on Computational Learning Theory,
        pages 287-294, 1992.

    [2] N. Abe and H. Mamitsuka. Query learning strategies using boosting and bagging.
        In Proceedings of the International Conference on Machine Learning (ICML),
        pages 1-9. Morgan Kaufmann, 1998.
    """

    def __init__(self, X=None, y=None, seed=None, disagreement='vote_entropy'):
        super(QBC, self).__init__(X, y, seed)
        self.disagreement = disagreement
        self.name = 'qbc'

        if disagreement in ['vote_entropy', 'KL_divergence']:
            self._disagreement = disagreement
        else:
            raise ValueError('Disagreement must be one of ["vote_entropy", "KL_divergence"]')
        
        
    def select_batch(self, model, labeled, N, n_jobs=None, **kwargs):
        """Select indexes from the unlabeled for querying.

        Parameters
        ----------
        labeled: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        model: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.

        N: int, optional (default=1)
            Selection batch size.

        n_jobs: int, optional (default=None)
            How many threads will be used in training bagging.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabeled.
        """
        
        all_inds = np.arange(self.X.shape[0])
        labeled = np.asarray(labeled)
        unlabeled = np.setdiff1d(all_inds, labeled)

        X_unlab = self.X[unlabeled]
        X_lab = self.X[labeled]
        y_lab = self.y[labeled]
        #####################################

        # bagging
        if isinstance(model, ScikitModel):
            bagging = SkBagging(model, n_jobs=4, n_estimators=10)
            bagging.fit(X_lab, y_lab)
        elif isinstance(model, TorchModel):            
            train_data = TensorDataset(torch.from_numpy(X_lab), torch.from_numpy(y_lab).flatten())
            train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

            bagging = TorchBagging(model, n_estimators=3, n_jobs=2, cuda=False)
            bagging.set_optimizer("Adam", lr=1e-3, weight_decay=5e-4)
            toc = time.time()
            bagging.fit(train_loader, epochs=40)
            print("training_time =" + str(time.time() - toc))
        else:
            raise TypeError("Only Scikit or Torch models allowed")
        
        estimators = bagging.estimators_

        # calc score
        if self._disagreement == 'vote_entropy':
            if isinstance(model, ScikitModel):
                score = self.calc_vote_entropy([estimator.predict(X_unlab) for estimator in estimators])
            elif isinstance(model, TorchModel):
                score = self.calc_vote_entropy([estimator.predict_numpy(torch.from_numpy(X_unlab)) for estimator in estimators])
        else:
            score = self.calc_avg_KL_divergence([estimator.predict_proba(X_unlab) for estimator in estimators])
        
        top_n = np.argpartition(score, -N)[-N:]
        return unlabeled[top_n]

    @classmethod
    def calc_vote_entropy(cls, predict_matrices):
        """Calculate the vote entropy for measuring the level of disagreement in QBC.

        Parameters
        ----------
        predict_matrices: list
            The prediction matrix for each committee.
            Each committee predict matrix should have the shape [n_samples, n_classes] for probabilistic output
            or [n_samples] for class output.

        Returns
        -------
        score: list
            Score for each instance. Shape [n_samples]

        References
        ----------
        [1] I. Dagan and S. Engelson. Committee-based sampling for training probabilistic
            classifiers. In Proceedings of the International Conference on Machine
            Learning (ICML), pages 150-157. Morgan Kaufmann, 1995.
        """
        score = []
        input_shape, committee_size = QBC._check_committee_results(predict_matrices)
        if len(input_shape) == 2:
            ele_uni = np.unique(predict_matrices)
            if not (len(ele_uni) == 2 and 0 in ele_uni and 1 in ele_uni):
                raise ValueError("The predicted label matrix must only contain 0 and 1")
            # calc each instance
            for i in range(input_shape[0]):
                instance_mat = np.array([X[i, :] for X in predict_matrices if X is not None])
                voting = np.sum(instance_mat, axis=0)
                tmp = 0
                # calc each label
                for vote in voting:
                    if vote != 0:
                        tmp += vote / len(predict_matrices) * np.log(vote / len(predict_matrices))
                score.append(-tmp)
        else:
            input_mat = np.array([X for X in predict_matrices if X is not None])
            # label_arr = np.unique(input_mat)
            # calc each instance's score
            for i in range(input_shape[0]):
                count_dict = collections.Counter(input_mat[:, i])
                tmp = 0
                for key in count_dict:
                    tmp += count_dict[key] / committee_size * np.log(count_dict[key] / committee_size)
                score.append(-tmp)
        return score

    @classmethod
    def calc_avg_KL_divergence(cls, predict_matrices):
        """Calculate the average Kullback-Leibler (KL) divergence for measuring the
        level of disagreement in QBC.

        Parameters
        ----------
        predict_matrices: list
            The prediction matrix for each committee.
            Each committee predict matrix should have the shape [n_samples, n_classes] for probabilistic output
            or [n_samples] for class output.

        Returns
        -------
        score: list
            Score for each instance. Shape [n_samples]

        References
        ----------
        [1] A. McCallum and K. Nigam. Employing EM in pool-based active learning for
            text classification. In Proceedings of the International Conference on Machine
            Learning (ICML), pages 359-367. Morgan Kaufmann, 1998.
        """
        score = []
        input_shape, committee_size = QBC._check_committee_results(predict_matrices)
        if len(input_shape) == 2:
            label_num = input_shape[1]
            # calc kl div for each instance
            for i in range(input_shape[0]):
                instance_mat = np.array([X[i, :] for X in predict_matrices if X is not None])
                tmp = 0
                # calc each label
                for lab in range(label_num):
                    committee_consensus = np.sum(instance_mat[:, lab]) / committee_size
                    for committee in range(committee_size):
                        tmp += instance_mat[committee, lab] * np.log(instance_mat[committee, lab] / committee_consensus)
                score.append(tmp)
        else:
            raise Exception(
                "A 2D probabilistic prediction matrix must be provided, with the shape like [n_samples, n_class]")
        return score
    
    @staticmethod
    def _check_committee_results(predict_matrices):
        """check the validity of given committee predictions.

        Parameters
        ----------
        predict_matrices: list
            The prediction matrix for each committee.
            Each committee predict matrix should have the shape [n_samples, n_classes] for probabilistic output
            or [n_samples] for class output.

        Returns
        -------
        input_shape: tuple
            The shape of the predict_matrix

        committee_size: int
            The number of committees.

        """
        shapes = [np.shape(X) for X in predict_matrices if X is not None]
        uniques = np.unique(shapes, axis=0)
        if len(uniques) > 1:
            raise Exception("Found input variables with inconsistent numbers of"
                            " shapes: %r" % [int(l) for l in shapes])
        committee_size = len(predict_matrices)
        if not committee_size > 1:
            raise ValueError("Two or more committees are expected, but received: %d" % committee_size)
        input_shape = uniques[0]
        return input_shape, committee_size

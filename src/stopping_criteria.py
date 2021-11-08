import abc
import math
import utils
import numpy as np

# RESULT_PATH = al_config.RESULTS + '/test/'

class StoppingCriterion(abc.ABC):

    @abc.abstractmethod
    def __init__(self, train_size=None, batch_size=None,
                 seed_batch=None, X=None, y=None):
        self.train_size = train_size
        self.batch_size = batch_size
        self.seed_batch = seed_batch
        self.X = X
        self.y = y
        self.annotated_total = 0
        self.current_batch = 0


    def next_state(self, annotated_count):
        self.annotated_total += annotated_count
        self.current_batch += 1


    @abc.abstractmethod
    def is_over(self, **kwargs):
        pass


    @abc.abstractmethod
    def reset(self):
        pass



class ConfidenceDrop(StoppingCriterion):

    def __init__(self, train_size, batch_size, seed_batch,
                 X, y, multilabel, train_horizon=1.0):
        super(ConfidenceDrop, self).__init__(train_size, batch_size, seed_batch, X, y)
        self.n_batches = 1 + int(math.ceil((train_horizon*train_size - seed_batch) / batch_size))
        self.confidence = []
        self.labeled = []
        self.multilabel = multilabel
 

    def is_over(self, model, batch_selected_inds, selected_inds, **kwargs):
        if len(selected_inds) == self.X.shape[0]:
            result = dict(labeled=self.labeled, conf=self.confidence)
            filepath = RESULT_PATH + f'conf.{utils.time_string()}.json'
            utils.dump(result, filepath)
            return True

        if utils.check_fitted(model):
            mask = np.ones(self.X.shape[0], np.bool)
            mask[selected_inds] = 0
            Xs = self.X[mask]

            self.labeled.append(len(selected_inds))
            probs = model.predict_proba(Xs)
            if self.multilabel:
                confidence = np.mean(probs[np.where(probs > 0.5)])
            else:
                confidence = np.mean(np.max(probs, axis=1))

            self.confidence.append(confidence)
            
        return False


    def reset(self):
        self.current_batch = 0



# class SeparatedConfidenceDrop(StoppingCriterion):

#     def __init__(self, train_size, batch_size, seed_batch,
#                  X, y, multilabel, drops_limit = 10,
#                  train_horizon=1.0):
#         super(SeparatedConfidenceDrop, self).__init__(train_size, batch_size, seed_batch, X, y)
#         self.n_batches = 1 + int(math.ceil((train_horizon*train_size - seed_batch) / batch_size))
#         self.confidence = []
#         self.labeled = []
#         self.multilabel = multilabel
#         self.prev_conf = None
#         self.consecutive_drops = 0
#         self.drops_limit = drops_limit
 

#     def is_over(self, model, batch_selected_inds, selected_inds, **kwargs):
#         if self.current_batch >= self.n_batches:
#             result = dict(labeled=self.labeled, conf=self.confidence)
#             filepath = RESULT_PATH + f'conf.{utils.time_string()}.json'
#             utils.dump(result, filepath)
#             return True

#         if utils.check_fitted(model):
#             mask = np.ones(self.X.shape[0], np.bool)
#             mask[selected_inds] = 0
#             Xs = self.X[mask]

#             self.labeled.append(len(selected_inds))
#             probs = model.predict_proba(Xs)+
#             if self.multilabel:
#                 confidence = np.mean(probs[np.where(probs > 0.5)])
#             else:
#                 confidence = np.mean(np.max(probs, axis=1))

#             self.confidence.append(confidence)
#             if self.prev_conf is not None:
#                 if confidence <= self.prev_conf:
#                     print("DROPPED")
#                     self.consecutive_drops += 1
#                 else:
#                     self.consecutive_drops = 0
            
#             self.prev_conf = confidence
#             return self.consecutive_drops >= self.drops_limit


#         return False


#     def reset(self):
#         self.current_batch = 0


class VarianceSurge(StoppingCriterion):

    def __init__(self, train_size, batch_size, seed_batch,
                 X, y, threshold = 5e-3, train_horizon=1.0):
        super(VarianceSurge, self).__init__(train_size, batch_size, seed_batch, X, y)
        self.n_batches = 1 + int(math.ceil((train_horizon*train_size - seed_batch) / batch_size))
        self.vars = []
        self.labeled = []
        self.threshold = threshold
 

    def is_over(self, model, batch_selected_inds, selected_inds, **kwargs):
        if utils.check_fitted(model) and len(batch_selected_inds) > 0:
            Xs, ys = self.X[batch_selected_inds], self.y[batch_selected_inds]
            self.labeled.append(len(selected_inds))
            probs = model.predict_proba(Xs)
            confidence = np.max(probs, axis=1)
            var = np.var(confidence)
            self.vars.append(var)
            return var > self.threshold

        return False


    def reset(self):
        self.current_batch = 0


class QueryCount(StoppingCriterion):

    def __init__(self, train_size, batch_size, seed_batch, train_horizon=1.0):
        super(QueryCount, self).__init__(train_size, batch_size, seed_batch)
        self.n_batches = 1 + int(math.ceil(0.5 * (train_horizon*train_size - seed_batch) / batch_size) )
 

    def is_over(self, **kwargs):
        return self.current_batch >= self.n_batches


    def reset(self):
        self.current_batch = 0



class UsersChoice(StoppingCriterion):

    def __init__(self):
        super(UsersChoice, self).__init__()


    def is_over(self):
        return False


    def reset(self):
        self.current_batch = 0

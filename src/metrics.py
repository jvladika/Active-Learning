import functools as ft

from sklearn import metrics
from collections import namedtuple

import numpy as np

MetricsModel = namedtuple(
    'Metrics', 'aggregated per_category')


class Metrics:

    def __init__(self, binarizer=None):
        if binarizer:
            self.binarizer = binarizer
            self.n_classes = len(binarizer.classes_)


        self.single_metrics = dict([
            ('f1_macro', ft.partial(metrics.f1_score, average='macro')),
            ('f1_micro', ft.partial(metrics.f1_score, average='micro')),
        ])

        self.category_metrics = dict([
            ('f1_binary', ft.partial(metrics.f1_score, average=None)),
            ('confusion_matrix', metrics.multilabel_confusion_matrix)
        ])

    def eval_metrics(self, ys_true, hs):
        return MetricsModel(
            dict([(name, metric(ys_true, hs))
                  for name, metric in self.single_metrics.items()]),
            dict([(name, metric(ys_true, hs))
                  for name, metric in self.category_metrics.items()]),
        )

    def eval_single_metric(self, ys_true, hs):
        return self.single_metrics['f1_macro'](ys_true, hs)

    def eval_per_category(self, ys_true, hs):
        if not self.binarizer:
            raise Exception('Cannot call this method without a binarizer')
        iteration = enumerate(self.binarizer.classes_)
        for i, category in iteration:
            h_category = hs[:, i]
            y_category = ys_true[:, i]
            evaluated_metrics = [
                (name, metric(y_category, h_category))
                for name, metric in self.category_metrics.items()]
            yield category, evaluated_metrics



def safe_div(nom, denom):
    a = np.array(nom)
    b = np.array(denom)
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a,b)
        c[c == np.inf] = 0
        c = np.nan_to_num(c)
    return c


def performance_measures(CM, average='micro'):
    tp = CM[:, 1, 1]
    tn = CM[:, 0, 0]
    fp = CM[:, 0, 1]
    fn = CM[:, 1, 0]

    tp_sum = tp.sum()
    tn_sum = tn.sum()
    fp_sum = fp.sum()
    fn_sum = fn.sum()
    ACC = (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum)

    if average=='micro':
        pred_sum = tp_sum + fp_sum
        true_sum = tp_sum + tn_sum
        P = safe_div(tp_sum, (tp_sum + fp_sum))
        R = safe_div(tp_sum, (tp_sum + fn_sum))
        F1 = safe_div(2 * P * R, (P + R))

    elif average=='macro':
        Ps = safe_div(tp, (tp + fp))
        Rs = safe_div(tp, (tp + fn))
        F1s = safe_div(2 * Ps * Rs, (Ps + Rs))
        P = Ps.mean()
        R = Rs.mean()
        F1 = F1s.mean()

    elif average=='weighted':
        Ps = safe_div(tp, (tp + fp))
        Rs = safe_div(tp, (tp + fn))
        F1s = safe_div(2 * Ps * Rs, (Ps + Rs))
        weights = tp + fn
        P = np.average(Ps, weights=weights)
        R = np.average(Rs, weights=weights)
        F1 = safe_div(2 * P * R, (P + R))

    else:
        raise ValueError('Average method ' + average + ' is not supported.')

    return P, R, F1, ACC
    

metric_info = dict(precision=(0,'precision'), recall=(1,'recall'), f1=(2,'f1'), accuracy=(3,'accuracy'))
f1_micro = lambda r: r['metrics_test']['aggregated']['f1_micro'], 'f1_micro'
f1_macro = lambda r: r['metrics_test']['aggregated']['f1_macro'], 'f1_macro'
conf_mat = lambda r: np.array(r['metrics_test']['per_category']['confusion_matrix'])


def get_metric(metric, average):
    info = metric_info[metric] 
    name = info[1] + '_' + average
    def calculate(r):
        CM = conf_mat(r)
        result = performance_measures(CM, average=average)
        return result[info[0]]
    return calculate, name


def area_metric(xs, ys):
    return np.sum(np.array(ys) / np.array(xs))
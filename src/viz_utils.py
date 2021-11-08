import json
import os, sys
import constants
import utils
import numpy as np
import matplotlib.pyplot as plt
import mpld3
from collections import defaultdict
import itertools
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import classification_report
from collections import Counter
import pickle


get_dataset = lambda x: x['meta']['dataset'], 'dataset'
get_warmstart_size = lambda x: x['meta']['warmstart_size'], 'warm'
get_batch_size = lambda x: x['meta']['batch_size'], 'batch'
get_model = lambda x: x['meta']['model'], 'model'
get_sampler = lambda x: x['meta']['sampler'], 'sampler'

getters = [get_dataset, get_warmstart_size, get_batch_size, get_model, get_sampler]


def gather_group(*args, path, datasets=None, filt=None):
    evals = defaultdict(list)
    group_getters = [get for get in getters if get not in args] # filter
    for file in os.listdir(path):
        if "grid=-" not in file:
            continue
        cont_flag = False
        if filt is not None:
            for f in filt:
                if f not in file:
                    cont_flag = True
                    break
        if cont_flag: continue
        if file.endswith('.json') and 'avg' in file:
            with open(os.path.join(path, file)) as f:
                r = json.load(f)
                if datasets is not None:
                    if get_dataset[0](r) not in datasets: continue
            info = tuple([(get[0](r), get[1]) for get in group_getters])
            evals[info].append(r)
    return evals


f1_micro = lambda r: r['f1_micro'], 'f1_micro', lambda r: r['f1_micro_baseline'], lambda r: r['f1_micro_var']
f1_macro = lambda r: r['f1_macro'], 'f1_macro', lambda r: r['f1_macro_baseline'], lambda r: r['f1_macro_var']


def plot_variance(axes, metrics, r, lbl=None, info=None):
    results = r['results']
    xs = []
    Ys = [[] for _ in range(len(metrics))]
    for res in results:
        xs.append(res['labeled'])
        for j, metric in enumerate(metrics):
            val = metric[0](res)
            Ys[j].append(val)
    
    lines = []
    for ys, ax, metric in zip(Ys, axes, metrics):
        l, = ax.plot(xs, ys, linewidth=2)
        lines.append(l)
        
    if info is not None:
        areas = []
        for ys in Ys:
            areas.append(area_metric(xs, ys))
        info[lbl] = tuple(areas)
    
    return tuple(lines)


def plot_result_avg(axes, r, colors, linestyles, metrics=[f1_micro, f1_macro],lbl=None, info=None):
    results = r['results']
    xs = results['labeled']
    Ys = []
    for i in range(len(metrics)):
        Ys.append(metrics[i][0](results))
    
    lines = []
    i = 0
    for ys, ax, metric in zip(Ys, axes, metrics):
        l, = ax.plot(xs, ys, linewidth=2, color=colors[i], linestyle=linestyles[i])
        lines.append(l)
        i += 1
        
    if info is not None:
        areas = []
        for ys in Ys:
            areas.append(area_metric(xs, ys))
        info[lbl] = tuple(areas)
    
    return tuple(lines)


def plot_baseline(axes, r, metrics):
    results = r['results']
    n = results['labeled'][-1] + 1
    xs = [0, n]
    Ys = []
    
    for i, metric in enumerate(metrics):
        Ys.append(2*[metric[2](results)])
    
    lines = []
    for ys, ax, metric in zip(Ys, axes, metrics):
        l, = ax.plot(xs, ys, linewidth=2)
        lines.append(l)

    return tuple(lines)


def create_label(getters):
    def labeler(r):
        label = []
        s = ', '
        for g, info in getters:
            label.append(info + '=' + str(g(r)))
        return s.join(label)
    return labeler


def create_title(elems):
    title = []
    s = ', '
    for elem, info in elems:
        title.append(info + ': ' + str(elem))
    return s.join(title)


def create_graphs(evals, label, metrics, draw_baseline=True, verbose=False):
    info_lines = []
    for i, (k, vs) in enumerate(evals.items()):
        lines = []
        labels = []
       
        title = create_title(k)
        info_lines.append(title)
        info_lines.append('='*50)
        
        N = len(metrics)
        fig = plt.figure(figsize=(13,7*N))
        fig.subplots_adjust(right=0.75, left=0.05, hspace=0.1)
        axes = []
        
        for i, m in enumerate(metrics, 1):
            if m[1] == "f1_macro":
                met = "F1 Macro"
            else:
                continue
            ax = fig.add_subplot(N,1,i)
            axes.append(ax)
            if i==1:
                ax.set_title(title, size=18)
            ax.set_ylabel(met, size=14)
            ax.grid(alpha=0.2)
            
            if i==N:
                ax.set_xlabel('number of labeled points', size=14)

        info = dict()
        for v in vs:
            lbl = label(v)
            labels.append(lbl)
            
            clrs = []
            linestyles = []
            for lab in labels:
                if lab == 'random':
                    clrs.append('purple')
                    linestyles.append(':')
                elif lab == 'qbc':
                    clrs.append('blue')
                    linestyles.append('-')
                elif lab == 'bald':
                    clrs.append('green')
                    linestyles.append('-')
                elif lab == 'entropy':
                    clrs.append('red')
                    linestyles.append('-.')
                elif lab == 'margin':
                    clrs.append('orange')
                    linestyles.append('-.')
                elif lab == 'least_confident':
                    clrs.append('yellow')
                    linestyles.append('-.')
            
            l = plot_result_avg(axes, v, clrs, linestyles, metrics)
            
            lines.append(l)
        
        if draw_baseline:
            l = plot_baseline(axes, v, metrics)
            lines.append(l)
            labels.append('baseline')
        
        
        sorts = []
        for i in range(len(metrics)):
            sort = sorted(info.items(), key=lambda kv: kv[1][i], reverse=True)
            info_lines.append('')
            info_lines.append('-'*20 + metrics[i][1] + '-'*20)
            for j, (key, val) in enumerate(sort, 1):
                string = '%2d %20s %10f' % (j, key, val[0])
                info_lines.append(string)
        
        info_lines.append('')
        legend = mpld3.plugins.InteractiveLegendPlugin(lines, labels, alpha_unsel=0., alpha_over=3., ax=axes[0])
        mpld3.plugins.connect(fig, legend)
    
    if verbose: print('\n'.join(info_lines))
    
    
def display_graphs(path, datasets=None, mode='normal', label=get_sampler[0], group=[get_sampler],
                   metrics=[f1_macro, f1_micro], draw_baseline=True, filt=None, verbose=False):
    evals = gather_group(*group, path=path, datasets=datasets, filt=filt)
    create_graphs(evals=evals, label=label, metrics=metrics, draw_baseline=draw_baseline, verbose=verbose)


def load_data(filepath):
    return utils.load(filepath, mode='pickle')


def convert_indices(y):
    classes = np.unique(y, axis=0)
    mapper = {}
      
    index = 0
    for c in classes:
        mapper[tuple(c)] = index
        index+=1
        
    y = [mapper[tuple(i)] for i in y]
    return np.array(y)
        

def count_multilabel(Y):
    c = Counter()
    for y in Y:
        indices, = np.nonzero(y)
        c.update(indices)
        
    return zip(*c.items())


def plot_class_distribution(filepath, multilabel):
    data = load_data(filepath)
    fig, axes = plt.subplots(1,2,figsize=(13,4))
    x_train, x_test, y_train, y_test = data
    if multilabel:
        l_train, v_train = count_multilabel(y_train)
        l_test, v_test = count_multilabel(y_test)
   
    l_train, v_train = zip(*Counter(y_train).items())
    l_test, v_test = zip(*Counter(y_test).items())

    plt.setp(axes, xticks=l_train)
    axes[0].set_title('Class distribution')
    axes[0].bar(l_train, v_train, label='train')
    axes[1].set_title('Class distribution')
    axes[1].bar(l_test, v_test, label='test', color='orange')
    axes[0].legend()
    axes[1].legend()


def load(path, mode='json'):
    """The method is used to load data. The supported formats are json, and
    pickle. To support other formats just add a loader (must have load() function implemented)
    into loaders, and the specified mode for opening the file in load_modes.

    Parameters
    ----------
    path : path
        path to stored data
    mode : str, optional
        method to use when loading the data, by default 'json'
        json and pickle supported for now

    Returns
    -------
    loaded data

    Raises
    ------
    ValueError
        if path doesn't exist
    """
    loaders = {'json': json.load, 'pickle': partial(
        pickle.load, encoding='latin1')}
    load_modes = {'json': 'r', 'pickle': 'rb'}

    if not os.path.exists(path):
        raise ValueError('The specified load path : %s doesn\'t exist' % path)
    with open(path, load_modes[mode]) as f:
        return loaders[mode](f)
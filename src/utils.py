import json
import os
import pickle
import collections
import numpy as np
import torch
import spacy

from glob import glob
from functools import partial
from time import time, strftime, gmtime, localtime
from datetime import datetime, timedelta
from datasets import load_dataset

import logging
import sys

import al_config

from podium.datasets.hf import HFDatasetConverter
from podium.datasets.impl import SST, IMDB
from podium import (Field, LabelField, Vocab, Example, TabularDataset)


def strip_extension(path):
    """Strips the extension from the path.

    Parameters
    ----------
    path : path
         path to strip the extension from

    Returns
    -------
    path : 
        path with no extension
    """
    idx = path.rfind('.')
    return path if idx < 0 else path[:idx]


def filename(path):
    """Return the filename from the full path.

    Parameters
    ----------
    path : path
        path to extract filename from

    Returns
    -------
    filename : string
        filename
    """
    return path.split('/')[-1]


def versioned_path(fullpath, sub='*', date_only=False):
    """Creates a versioned path by replacing the sub symbol with a timestamp string.

    Parameters
    ----------
    fullpath : path
        path to be versioned
    sub : str, optional
        sub symbol, will be replaced with a timestamp, by default '*'
    date_only : bool, optional
        whether or not to use only date in the timestamp, by default False

    Returns
    -------
    path:
        versioned path

    Raises
    ------
    ValueError
        if fullpath contains more than one or no sub symbols
    """

    if fullpath.count(sub) != 1:
        raise Exception('Use exactly one substitution symbol')

    return fullpath.replace(sub, '%s') % timestr(date_only=date_only)


def versioned_last(fullpath):
    """Returns the last versioned file from a directory given with path=fullpath.

    Parameters
    ----------
    fullpath : path
        path to a directory containing versioned files

    Returns
    -------
    path
        path to the last versioned file in the directory specified with fullpath

    Raises
    ------
    ValueError
        if the fullpath directory doensn't exist
    """

    if not os.path.exists(fullpath):
        raise ValueError('The given path: %s doesn\'t exist' % fullpath)

    files = glob(fullpath)
    files.sort(key=os.path.getmtime, reverse=False)

    # return newest file
    return files[-1]


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


def dump(obj, path, overwrite=False, mode='json'):
    """The method is used to store data. The supported formats are json, and
    pickle. To support other formats just add a dump method (dumper) (must have dump() function implemented)
    into dumpers, and the specified mode for opening the file in dump_modes.

    Parameters
    ----------
    obj:
        object to dump
    path : path
        path where to store the data
    overwrite: boolean
        whether to overwrite if the specified path already exist
    mode : str, optional
        method to use when dumping the data, by default 'json'
        json and pickle supported for now

    Raises
    ------
    Exception
        if specified path already exists but shouldn't be overwritten
    """

    if not os.path.exists(al_config.RESULTS):
        os.makedirs(al_config.RESULTS)

    dumpers = {'json': partial(json.dump, indent=2, cls=CommonObjectEncoder),
              'pickle': partial(pickle.dump, protocol=pickle.HIGHEST_PROTOCOL)}
    dump_modes = {'json': 'w', 'pickle': 'wb'}

    if os.path.exists(path) and not overwrite:
        raise Exception('Path %s exists. Will not overwrite.' % path)
    with open(path, dump_modes[mode]) as f:
        if mode=='json':
            obj = _namedtuple_asdict(obj)
        dumpers[mode](obj, f)


class CommonObjectEncoder(json.JSONEncoder):
    '''
    JSON encode numpy arrays
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def _namedtuple_asdict(obj):
    '''
    Encode namedtuples.

    Source adapted from:
    https://stackoverflow.com/questions/16938456/
    serializing-a-nested-namedtuple-into-json-with-python-2-7
    '''
    recurse = lambda x: map(_namedtuple_asdict, x)
    obj_is = lambda x: isinstance(obj, x)
    if obj_is(tuple) and hasattr(obj, '_fields'):
        fields = zip(obj._fields, recurse(obj))
        class_name = obj.__class__.__name__
        return dict(fields, **{'_type': class_name})
    elif obj_is(collections.Mapping):
        return type(obj)(zip(obj.keys(), recurse(obj.values())))
    elif obj_is(collections.Iterable) and not (
            obj_is(str) or obj_is(np.ndarray)):
        return type(obj)(recurse(obj))
    else:
        return obj



def logger(name):
    """
    Basic logger configured to write to stderr and to a file.

    Parameters
    ----------
    name : string
        program name used for naming the log file
    """
    if not os.path.exists(al_config.LOG):
        os.makedirs(al_config.LOG)
    log_file = os.path.join(al_config.LOG, f'{name}.{timestr()}.log')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = [logging.StreamHandler(sys.stderr), logging.FileHandler(log_file)]
    return logger


def time_string():
    fmt = '%Y-%m-%d-%H-%M'
    now = datetime.now() + timedelta(hours=1)
    return now.strftime(fmt)


def timestr(date_only=False):
    """
    Parameters
    ----------
    date_only : bool, optional
        [description], by default False
    """
    fmt = '%Y-%m-%d' + ('' if date_only else '-%H-%M-%S')
    return strftime(fmt, gmtime())


def stratified_sample(y, total):
	"""
	Arguments:
		y - data
		total - sample size

	Returns a stratified sample from 'y', of length 'total'.
	Method will result in total // number_of_classes examples per class.
	The remaining 'r' examples to 'total' will be drawn from the first 'r' classes.
	"""

	per_class = lambda nc, tot: (1, 0) if tot == -1 else (int(tot//nc), tot%nc)

	if len(y.shape) == 1:
		# non-multilabel data
		num_classes = y.max() + 1
		num_pc, remainder = per_class(num_classes, total)
		indices = [np.where(y == i)[0] for i in range(num_classes)]
	else:
		# multilabel data
		num_classes = y.shape[1]
		num_pc, remainder = per_class(num_classes, total)
		indices = [np.where(y[:,i] == 1)[0] for i in range(num_classes)]

	selected_inds = []
	for ind_set in indices:
			count = num_pc
			if remainder > 0:
				count += 1
				remainder -= 1
			indices = np.random.choice(ind_set, count, replace=False)
			selected_inds.extend(indices)

	return selected_inds   


def convert_indices(y):
		classes = np.unique(y, axis=0)
		mapper = {}

		index = 0
		for c in classes:
				mapper[tuple(c)] = index
				index += 1

		y = [mapper[tuple(i)] for i in y]
		return np.array(y)


def check_fitted(clf): 
    return hasattr(clf, 'classes_')


def save_data(data, filepath, id, save_raw, save_id):
    entries = []
    for example in data.examples:
        entry = dict(text=example.text[1],
                     label=example.label[1])
        if save_raw:
            entry['raw'] = example.text[0]
        if save_id:
            entry['id'] = id
        entries.append(json.dumps(entry))
        id += 1

    json_dicts = '\n'.join(entries)
    with open(filepath, 'w') as f:
        f.write(json_dicts)
    print(f'Saved data at {filepath}.')
    return id


def save_dataset(dataset, path,
                 save_raw=True, save_id=True):
    id = 0
    filepath = os.path.join(path, f'test.json')
    id = save_data(dataset, filepath, id, save_raw, save_id)


DATASETS = {
    'sst': SST,
    'imdb': IMDB
}


def load_data(dataset, max_vocab_size=5_000, include_lengths=False, field_names=['label', 'text'], validation=False):
    
    vocab = Vocab(max_size=max_vocab_size)

    LABEL = LabelField(field_names[0])
    TEXT = Field(field_names[1], numericalizer=vocab, include_lengths=include_lengths, disable_batch_matrix=False)
    if len(field_names) > 2:
        TEXT2 = Field(field_names[2], numericalizer=vocab, include_lengths=include_lengths, disable_batch_matrix=False)
        fields = {'text1': TEXT, 'text2' : TEXT2, 'label': LABEL}
    else:
        fields = {'text': TEXT, 'label': LABEL}

    if dataset in DATASETS:
        dataset_cls = DATASETS[dataset]
        splits = dataset_cls.get_dataset_splits(fields=fields)
    else:
        if 'glue' in dataset:
            hf_dataset = load_dataset('glue', dataset.split(',')[1])
        else:
            hf_dataset = load_dataset(dataset)
        if validation:
            train, test = hf_dataset['train'], hf_dataset['validation']
        else:
            train, test = hf_dataset['train'], hf_dataset['test']
        train_converter = HFDatasetConverter(train, fields=fields)
        test_converter = HFDatasetConverter(test, fields=fields)
        ds_train, ds_test = train_converter.as_dataset(), test_converter.as_dataset()

        if len(field_names) > 2:
            for i in range(len(ds_train)):
                ds_train[i].sentence1[1].append('[SEP]')
                ds_train[i].sentence1[1].extend(ds_train[i].sentence2[1])
            
            for i in range(len(ds_test)):
                ds_test[i].sentence1[1].append('[SEP]')
                ds_test[i].sentence1[1].extend(ds_test[i].sentence2[1])

        splits = (ds_train, ds_test)
        
    return splits, fields, vocab


# def load_dataset_for_transformer(path, tokenizer, lower=False,
#                                  stop_words=None, load_raw=True,
#                                  load_id=True, float_label=True,
#                                  max_len=512):
    
#     postpro = lambda xs, _: [tokenizer.convert_tokens_to_ids(x[:max_len])
#                              for x in xs]

#     TEXT = Field(use_vocab=False,
#                  postprocessing=postpro,
#                  pad_token=tokenizer.pad_token_id,
#                  lower=lower,
#                  stop_words=stop_words)
#     label_type = torch.float if float_label else torch.long
#     LABEL = LabelField(dtype=label_type)
#     # RAW = RawField()
#     # ID = RawField()

#     fields = {'text': ('text', TEXT),
#               'label': ('label', LABEL)}

#     if load_raw:
#         fields['raw'] = ('raw', RAW)
#         RAW.is_target = True

#     if load_id:
#         fields['id'] = ('id', ID)
#         ID.is_target = True

#     splits = data.TabularDataset.splits(
#                                 path=path,
#                                 train='train.json',
#                                 validation='valid.json',
#                                 test='test.json',
#                                 format='json',
#                                 fields=fields)

#     return splits, (TEXT, LABEL, RAW, ID)


# def load_data_for_transformer(LOAD_PATH, tokenizer, lower=False,
#                               stop_words=None, load_raw=True,
#                               load_id=True, float_label=True,
#                               max_len=512):

#     splits, fields = load_dataset_for_transformer(LOAD_PATH,
#                                                   tokenizer,
#                                                   lower,
#                                                   stop_words,
#                                                   load_id,
#                                                   load_raw,
#                                                   float_label,
#                                                   max_len)
#     # LABEL
#     fields[1].build_vocab(splits[0])
#     return splits, fields


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def set_seed_everywhere(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

'''
with open('./data/TREC/test_500.label', 'r', encoding='latin-1') as file:
    with open('./data/TREC/test.csv', 'w') as out:
        out.write('text,label\n')

        text = file.readlines()
        for line in text:
            tokens = line.split()
            label = tokens[0].split(':')[0]
            rest = line[len(tokens[0])+1:]
            out.write(rest[:-3] + ',' + label + '\n')
'''


def load_trec():
    vocab = Vocab(max_size=10000)
    TEXT = Field('text', numericalizer=vocab, keep_raw=True, include_lengths=False, disable_batch_matrix=False)
    LABEL = LabelField('label')
    fields = {'text': TEXT, 'label': LABEL}

    dataset = TabularDataset('./data/TREC/train.csv', fields=fields, format='csv')

    dataset1 = TabularDataset('./data/TREC/test.csv', fields=fields, format='csv')

    return (dataset, dataset1), (TEXT, LABEL), vocab

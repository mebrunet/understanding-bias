'''
Train a Word2Vec embedding and save the result
'''

# Imports
import logging
import sys
from time import time
from os import path
import json

import numpy as np
from numpy import random as npr
from gensim.models.word2vec import Word2Vec

# Basic Setup
NYT_CORPUS = '/Users/mebrunet/Code/UofT/understanding-bias/corpora/nytselect.txt'
NYT_INDEX = '/Users/mebrunet/Code/UofT/understanding-bias/corpora/nytselect.meta.json'
PERT_MAP = '/Users/mebrunet/Code/UofT/understanding-bias/results/perturbations/C1-V15-W8-D200-R0.05-E150-B1/pert_map.json'


# Helpers
def get_arg(i, default=None):
    '''Helper to get command line arg or return a default'''
    arg = sys.argv[i] if len(sys.argv) > i else default
    if arg is None:
        raise TypeError('Missing required argument (position {}).'.format(i))
    return arg


def print_mem():
    pid = getpid()
    ppid = getppid()
    pmem = Process(pid).memory_info().rss / 2**30
    ppmem = Process(ppid).memory_info().rss / 2**30
    print('Memory:', pid, pmem, 'GB', ppmem, 'GB')


class TextCorpus(object):
    def __init__(self, textfile, indexfile, without=[], shuffle=False, seed=1):
        with open(indexfile, 'r') as f:
            metadata = json.load(f)

        self.index = metadata['index']
        self.textfile = textfile
        self.without = without   # Filter line number
        self.rs = npr.RandomState(seed)
        self.shuffled = shuffle

        # Build a document ordering
        self.doc_order = []
        skips = sorted(set(without), reverse=True)
        next_skip = skips.pop() - 1 if len(skips) > 0 else -1  # convert to 0-index
        for i in range(metadata['num_documents']):
            if i == next_skip:
                next_skip = skips.pop() - 1 if len(skips) > 0 else -1
            else:
                self.doc_order.append(i)

        if shuffle:
            self.rs.shuffle(self.doc_order)

    def __iter__(self):
        with open(self.textfile, 'r', encoding='utf-8') as f:
            for document_num in self.doc_order:
                f.seek(self.index[document_num]['byte'])
                text = f.readline().strip()
                yield text.split(' ')


if __name__ == '__main__':
    textfile = get_arg(1, NYT_CORPUS)
    indexfile = get_arg(2, NYT_INDEX)
    outdir = get_arg(3, 'embeddings')
    workers = int(get_arg(4, '4'))
    pert_type = get_arg(5, 'aggravate_10000_1')
    pert_map_file = get_arg(6, PERT_MAP)

    with open(pert_map_file) as f:
        pert_map = json.load(f)

    omit = pert_map[pert_type]

    # Logging
    logging.basicConfig(filename=path.join(outdir, '{}.log'.format(pert_type)),
                        format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    for seed in range(1, 4):
        params = {'size': 200, 'window': 8, 'min_count': 15, 'seed': seed}
        print('Embedding with params', params, flush=True)
        print('Omitting', pert_type, ' (', len(omit), 'documents )')
        sentences = TextCorpus(textfile, indexfile, shuffle=True, seed=seed,
                               without=omit)
        outfile = '-'.join([str(x) for x in params.values()])
        w2v = Word2Vec(sentences, negative=10, workers=workers, **params)
        print('Finished training, saving...')
        w2v.save(path.join(outdir, 'nyt_{}_{}.w2v'.format(pert_type, outfile)))
        del w2v

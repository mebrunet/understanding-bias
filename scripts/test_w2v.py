import json
import numpy as np
from numpy import random as npr
from gensim.models.word2vec import Word2Vec

# %cd scripts

from train_w2v import TextCorpus, NYT_CORPUS, NYT_INDEX, PERT_MAP

NUM_DOCS = 1412846

with open(PERT_MAP) as f:
    pert_map = json.load(f)

omit = pert_map['aggravate_10000_1']

assert len(omit) == 10_000

# Load and run basic checks
tc_og = TextCorpus(NYT_CORPUS, NYT_INDEX, shuffle=False, seed=1, without=[])
tc_shuf = TextCorpus(NYT_CORPUS, NYT_INDEX, shuffle=True, seed=1, without=[])
tc_omit = TextCorpus(NYT_CORPUS, NYT_INDEX, shuffle=False, seed=1, without=omit)
tc_so = TextCorpus(NYT_CORPUS, NYT_INDEX, shuffle=True, seed=1, without=omit)

assert len(tc_og.doc_order) == NUM_DOCS
assert len(tc_shuf.doc_order) == NUM_DOCS
assert len(tc_omit.doc_order) == NUM_DOCS - len(omit)
assert len(tc_so.doc_order) == NUM_DOCS - len(omit)

assert tc_og.doc_order[0:5] == list(range(5))
assert tc_shuf.doc_order[0:5] != list(range(5))
assert tc_omit.doc_order[0:5] == list(range(5))  # Asumes doc 0-4 not excluded
assert tc_so.doc_order[0:5] != list(range(5))


# Check equivalence
with open(NYT_CORPUS, 'r', encoding='utf-8') as f:
    i = 0
    it = tc_og.__iter__()
    for line in f:
        actual = line.split()
        tc = it.__next__()
        assert actual == tc
        i += 1
        if i % 100_000 == 0:
            print(i)

# Iterate once through with shuffle
i = 0
for doc in tc_so:
    i += 1

assert i == NUM_DOCS - len(omit)

# And again...
j = 0
for doc in tc_so:
    j += 1

assert j == NUM_DOCS - len(omit)

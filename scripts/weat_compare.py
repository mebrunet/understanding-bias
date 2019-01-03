import sys
from os import path
import json

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from numpy import random as npr
from gensim.models import KeyedVectors, Word2Vec

from scripts.eval_w2v import calc_effect_size

with open("results/weat_compare/rand_weat_sets.json") as f:
    rand_weat_sets = json.load(f)

models = []
for i in range(1, 6):
    models.append(Word2Vec.load("results/w2v/nyt_baseline_200-8-15-{}.w2v".format(i)))

effect_sizes = []
for word_set in rand_weat_sets:
    temp = []
    for w2v in models:
        es = calc_effect_size(word_set["S"], word_set["T"],
                              word_set["A"], word_set["B"], w2v.wv)
        temp.append(es)

    mean_es = np.mean(temp)
    assert mean_es != es  # assert mean is different than last
    effect_sizes.append(mean_es)


df = pd.read_csv("results/weat_compare/glove_weat.csv")
df = df.join(pd.read_csv("results/weat_compare/cooc_weat.csv"))
df = df.join(pd.read_csv("results/weat_compare/ppmi_weat.csv"))
df["w2v"] = effect_sizes


plt.scatter(df["w2v"], df["glove"])
plt.show()

np.corrcoef(df["w2v"], df["glove"])
np.corrcoef(df["w2v"], df["ppmi"])
np.corrcoef(df["glove"], df["ppmi"])

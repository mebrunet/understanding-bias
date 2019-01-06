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
df.to_csv("results/weat_compare/combined.csv")

df = pd.read_csv("results/weat_compare/combined.csv")


# Plot
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

GRID_ALPHA = 0.25

# %% Word2Vec vs. GloVe
plt.scatter(df["glove"], df["w2v"], marker=".")
plt.ylabel("Word2Vec")
plt.xlabel("GloVe")
plt.show()
# %%


# %% PPMI vs. GloVe
fig = plt.figure(dpi=150, figsize=(6, 5))
plt.scatter(df["glove"], df["ppmi"], marker=".")
plt.ylabel("PPMI WEAT")
plt.xlabel("GloVe WEAT")
ax = plt.gca()
ax.set_aspect('equal')
XMIN, XMAX = plt.xlim((-2, 2))
YMIN, YMAX = plt.ylim((-2, 2))
S = 2 # tick scale
ax.set_xticks(np.arange(np.ceil(S*XMIN)/S, (np.floor(S*XMAX) + 1)/S, 1/S))
ax.set_yticks(np.arange(np.ceil(S*YMIN)/S, (np.floor(S*YMAX) + 1)/S, 1/S))
plt.tight_layout()
plt.savefig("results/weat_compare/glove_ppmi_scatter.png")
plt.show()
np.corrcoef(df["glove"], df["ppmi"])
# %%


# %% word2vec vs. GloVe
fig = plt.figure(dpi=150, figsize=(6, 5))
plt.scatter(df["glove"], df["w2v"], marker=".")
plt.ylabel("word2vec WEAT")
plt.xlabel("GloVe WEAT")
ax = plt.gca()
ax.set_aspect('equal')
XMIN, XMAX = plt.xlim((-2, 2))
YMIN, YMAX = plt.ylim((-2, 2))
S = 2 # tick scale
ax.set_xticks(np.arange(np.ceil(S*XMIN)/S, (np.floor(S*XMAX) + 1)/S, 1/S))
ax.set_yticks(np.arange(np.ceil(S*YMIN)/S, (np.floor(S*YMAX) + 1)/S, 1/S))
plt.tight_layout()
plt.savefig("results/weat_compare/glove_w2v_scatter.png")
plt.show()
np.corrcoef(df["glove"], df["w2v"])
# %%

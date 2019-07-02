# %% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import re
from os import path
from scipy import stats

# %% plotting consts

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


# %% Load
TARGET = "C0-V15-W8-D75-R0.05-E300-B2"
PERT_DIR = "results/perturbations"
SAVE_DIR = "results/weighted_weat"
CORPUS = "NYT" if TARGET.startswith("C1") else "Wiki"
BIAS = TARGET[-1:]

df = pd.read_csv(path.join(PERT_DIR, TARGET, "weighted_change.csv"))
non_rand = df[df["pert_type"] != "random"]

means = non_rand.groupby(["pert_type", "pert_size"])["trueB̃", "logB̃", "propB̃", "polyB̃"].mean()
stds = non_rand.groupby(["pert_type", "pert_size"])["trueB̃", "logB̃", "propB̃", "polyB̃"].std()


# %% Plot
SIZES = [100, 300, 1000, 3000, 10_000] if CORPUS == "NYT" else [10, 30, 100, 300, 1000]
positions = [("aggravate", s) for s in SIZES[::-1]]
positions += [("baseline", 0)]
positions += [("correct", s) for s in SIZES]
D = 1.0
d = 0.2

plt.figure(figsize=(8,7))
for (pert_type, pert_size), vals in means.iterrows():
    std = stds.loc[(pert_type, pert_size)]
    baseline = pert_type == "baseline"
    sign = -D if pert_type == "aggravate" else D
    y = 0 if baseline else sign * (1 + SIZES.index(pert_size))
    # print(pert_type, pert_size, ":", y)

    plt.errorbar(vals["trueB̃"], y, xerr=std["trueB̃"], label=("base" if baseline else None), marker="d", color="k")
    plt.errorbar(vals["logB̃"], y-d, xerr=std["logB̃"], label=("log" if baseline else None), marker="d", color="b")
    plt.errorbar(vals["polyB̃"], y-3*d, xerr=std["polyB̃"], label=("poly" if baseline else None), marker="d", color="r")
    plt.errorbar(vals["propB̃"], y-2*d, xerr=std["propB̃"], label=("prop" if baseline else None), marker="d", color="g")

plt.yticks(np.arange(len(positions)) - len(positions) // 2, positions)
plt.legend()
plt.xlabel("Effect Size")
plt.title("{} - B{}".format(CORPUS, BIAS))
plt.tight_layout()
plt.savefig(path.join(SAVE_DIR, "weighted_compare_{}_{}.png".format(CORPUS, BIAS)))
plt.show()
# %%

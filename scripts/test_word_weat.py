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
# TARGETS = ["C0-V15-W8-D75-R0.05-E300-B1"]
TARGETS = ["C0-V15-W8-D75-R0.05-E300-B1", "C0-V15-W8-D75-R0.05-E300-B2",
           "C1-V15-W8-D200-R0.05-E150-B1", "C1-V15-W8-D200-R0.05-E150-B2"]
PERT_DIR = "results/perturbations"
SAVE_DIR = "results/test_word_weat"
# %%

for TARGET in TARGETS:
    CORPUS = "NYT" if TARGET.startswith("C1") else "Wiki"
    BIAS = TARGET[-1:]
    # %%
    test_df = pd.read_csv(path.join(PERT_DIR, TARGET, "test_word_change.csv"))
    test_non_rand = test_df[test_df["pert_type"] != "random"]
    test_means = test_non_rand.groupby(["pert_type", "pert_size"])["trueB̃",].mean()
    test_stds = test_non_rand.groupby(["pert_type", "pert_size"])["trueB̃",].std()


    true_df = pd.read_csv(path.join(PERT_DIR, TARGET, "true_change_with_baselines.csv"))
    true_non_rand = true_df[true_df["pert_type"] != "random"]
    true_means = true_non_rand.groupby(["pert_type", "pert_size"])["trueB̃",].mean()
    true_stds = true_non_rand.groupby(["pert_type", "pert_size"])["trueB̃",].std()


    # %% Plot
    SIZES = [100, 300, 1000, 3000, 10_000] if CORPUS == "NYT" else [10, 30, 100, 300, 1000]
    positions = [("aggravate", s) for s in SIZES[::-1]]
    positions += [("baseline", 0)]
    positions += [("correct", s) for s in SIZES]
    D = 1.0
    d = 0.2

    plt.figure(figsize=(8,7))
    for (pert_type, pert_size), test_vals in test_means.iterrows():
        test_std = test_stds.loc[(pert_type, pert_size)]
        baseline = pert_type == "baseline"
        sign = -D if pert_type == "aggravate" else D
        y = 0 if baseline else sign * (1 + SIZES.index(pert_size))
        # print(pert_type, pert_size, ":", y)

        true_val = true_means.loc[(pert_type, pert_size)]
        true_std = true_stds.loc[(pert_type, pert_size)]

        plt.errorbar(true_val["trueB̃"], y, xerr=true_std["trueB̃"], label=("original" if baseline else None), marker="d", color="k")

        plt.errorbar(test_vals["trueB̃"], y, xerr=test_std["trueB̃"], label=("test" if baseline else None), marker="d", color="b")

    plt.yticks(np.arange(len(positions)) - len(positions) // 2, positions)
    plt.legend()
    plt.xlabel("Effect Size")
    plt.title("{} - B{}".format(CORPUS, BIAS))
    plt.tight_layout()
    plt.savefig(path.join(SAVE_DIR, "test_compare_{}_{}.png".format(CORPUS, BIAS)))
    plt.show()

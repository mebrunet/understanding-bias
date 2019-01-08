import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import re
from os import path
from scipy import stats


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

raw_df = pd.read_csv("results/direct_bias/projections.csv")

agg_cols = [col for col in raw_df if col.startswith('agg')]
base_cols = [col for col in raw_df if col.startswith('base')]
cor_cols = [col for col in raw_df if col.startswith('cor')]

df = pd.DataFrame({"word": raw_df["word"], "count": raw_df["count"],
                "agg_μ": raw_df[agg_cols].mean(axis=1),
                "agg_σ": raw_df[agg_cols].std(axis=1),
                "base_μ": raw_df[base_cols].mean(axis=1),
                "base_σ": raw_df[base_cols].std(axis=1),
                "cor_μ": raw_df[cor_cols].mean(axis=1),
                "cor_σ": raw_df[cor_cols].std(axis=1)})



# %% Plot
D = 3  # vertical space between words
d = 0.8

fig = plt.figure(dpi=150, figsize=(6, 5))

plt.axvline(0, c="k", ls="dotted")

positions = []
labels = []
first_pass = True
for pos, vals in df.iterrows():
    y = D * pos
    positions.append(y)
    labels.append("{} ({}k)".format(vals["word"], vals["count"]))
    plt.errorbar(vals["base_μ"], y, xerr=vals["base_σ"], marker="s", c="k",
                label=first_pass and "base" or None)
    plt.errorbar(vals["agg_μ"], y, xerr=vals["agg_σ"], marker="v", c="g",
                label=first_pass and "cor." or None)
    plt.errorbar(vals["cor_μ"], y, xerr=vals["cor_σ"], marker="^", c="r",
                label=first_pass and "agg." or None)
    first_pass = False

plt.legend()
plt.yticks(positions, labels)
plt.ylabel("SCIENCE                    ARTS   ")
plt.xlabel("male <-- gender axis --> female")
plt.grid(True, which="major", axis="y", linestyle="dotted")
plt.tight_layout()
plt.savefig("results/direct_bias/direct_bias.png")
plt.show()



# %%
deltas = np.abs(df["cor_μ"] - df["agg_μ"])
np.corrcoef(np.log(df["count"]), deltas)

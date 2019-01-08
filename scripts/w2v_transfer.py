import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import re
from os import path
from scipy import stats


PERT_DIR = "results/perturbations/C1-V15-W8-D200-R0.05-E150-B1"
SAVE_DIR = "results/w2v"

# Combine GloVe data
preds = pd.read_csv(path.join(PERT_DIR, "predicted_change.csv"),
                    usecols=["pert_type", "pert_size", "seed", "B", "IFB̃"])

base_weat = preds[["seed", "B"]].groupby("seed").mean()["B"]

trues = pd.read_csv(path.join(PERT_DIR, "true_change.csv"),
                    usecols=["pert_type", "pert_size", "seed", "trueB̃"])

trues_grouped = trues.groupby(["pert_type", "pert_size"])["trueB̃"]

df = pd.DataFrame({"glove_μ": trues_grouped.mean(),
                   "glove_σ": trues_grouped.std()})

baseline = pd.DataFrame({"pert_type": ["baseline"], "pert_size": [0], "glove_μ": [np.mean(base_weat)], "glove_σ": [np.std(base_weat)]}).set_index(["pert_type", "pert_size"])

df = df.append(baseline)

df = df.iloc[df.index.get_level_values("pert_type") != "random"]


# Add Word2Vec data
w2v = pd.read_csv(path.join(SAVE_DIR, "w2v.csv"),
                 usecols=["pert_type", "pert_size", "seed", "effect_size"])


w2v_grouped = w2v.groupby(["pert_type", "pert_size"])["effect_size"]

df["w2v_μ"] = w2v_grouped.mean()
df["w2v_σ"] = w2v_grouped.std()


# Add PPMI data
ppmi = pd.read_csv(path.join(PERT_DIR, "ppmi_change.csv"))

df["ppmi"] = ppmi.set_index(["pert_type", "pert_size"])["ppmiB̃"]


df


# %% Plot
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


fig = plt.figure(dpi=150, figsize=(6, 5))


# Y positions, and spacing
SIZES = [100, 300, 1000, 3000, 10000]
positions = [("aggravate", s) for s in SIZES[::-1]]
positions += [("baseline", 0)]
positions += [("correct", s) for s in SIZES]
D = 4  # vertical space between scenarios
d = 1  # vertial space between types


first_pass = True
for (pert_type, pert_size), vals in df.iterrows():
    pos = D * positions.index((pert_type, pert_size))
    plt.errorbar(vals["glove_μ"], pos + d, c='k', marker="p", xerr=vals["glove_σ"],
                label=first_pass and "GloVe" or None)
    plt.scatter(vals["ppmi"], pos, c='b', marker="d",
                label=first_pass and "PPMI" or None)
    plt.errorbar(vals["w2v_μ"], pos - d, c='r', marker=".", xerr=vals["w2v_σ"],
                label=first_pass and "word2vec" or None)
    first_pass = False
    # if (pert_type == "baseline"):
    #     plt.axvline(vals["glove_μ"], c='k', ls="dotted")
    #     plt.axvline(vals["ppmi"], c='b', ls="dotted")
    #     plt.axvline(vals["w2v_μ"], c='r', ls="dotted")



scenarios = [name + "-" + str(size) for (name, size) in positions][::-1]
plt.yticks(D * np.arange(len(positions)), scenarios)

XLIM = plt.xlim((-2, 2))
S = 2
ax = plt.gca()
ax.set_xticks(np.arange(np.ceil(S*XLIM[0])/S, (np.floor(S*XLIM[1]) + 1)/S, 1/S))

plt.grid(True, which="major", axis="y", linestyle="dotted")
plt.legend()
plt.xlabel("WEAT effect size")
plt.title("Transfer Effects: NYT - WEAT 1")

plt.tight_layout()
plt.savefig(path.join(SAVE_DIR, "transfer_effects.png"))
plt.show()

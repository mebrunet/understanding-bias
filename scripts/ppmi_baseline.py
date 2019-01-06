import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import re
from os import path
from scipy import stats


OG_DIR = "results/perturbations/C0-V15-W8-D75-R0.05-E300-B1"
PPMI_DIR = "results/ppmi_perturbations/C0-V15-W8-D75-R0.05-E300-B1"
SAVE_DIR = "results/ppmi_perturbations/figures"

og_preds = pd.read_csv(path.join(OG_DIR, "predicted_change.csv"))
og_trues = pd.read_csv(path.join(OG_DIR, "true_change.csv"))
og_ppmi = pd.read_csv(path.join(OG_DIR, "ppmi_change.csv"))

preds = pd.read_csv(path.join(PPMI_DIR, "predicted_change.csv"))
trues = pd.read_csv(path.join(PPMI_DIR, "true_change.csv"))
ppmi = pd.read_csv(path.join(PPMI_DIR, "ppmi_change.csv"))

og_ppmi.set_index(["pert_type", "pert_size"], inplace=True)
ppmi.set_index(["pert_type", "pert_size"], inplace=True)
df = ppmi.join(og_ppmi, rsuffix="_og")


# %%
IFBs = []
trueBs = []
preds_grouped = preds.groupby(["pert_type", "pert_size"])["IFB̃"].mean()
trues_grouped = trues.groupby(["pert_type", "pert_size"])["trueB̃"].mean()

og_IFBs = []
og_trueBs = []
og_preds_grouped = og_preds.groupby(["pert_type", "pert_size"])["IFB̃"].mean()
og_trues_grouped = og_trues.groupby(["pert_type", "pert_size"])["trueB̃"].mean()


for ((pert_type, pert_size), _) in df.iterrows():
    print(pert_type, pert_size)
    if pert_type == "baseline":
        IFBs.append(np.mean(np.unique(preds["B"])))
        trueBs.append(np.mean(np.unique(preds["B"])))  # baseline only in preds
        og_IFBs.append(np.mean(np.unique(og_preds["B"])))
        og_trueBs.append(np.mean(np.unique(og_preds["B"])))
    else:
        IFBs.append(preds_grouped[(pert_type, pert_size)])
        trueBs.append(trues_grouped[(pert_type, pert_size)])
        og_IFBs.append(og_preds_grouped[(pert_type, pert_size)])
        og_trueBs.append(og_trues_grouped[(pert_type, pert_size)])


df["IFB̃"] = IFBs
df["trueB̃"] = trueBs
df["IFB̃_og"] = og_IFBs
df["trueB̃_og"] = og_trueBs

df.to_csv("results/ppmi_perturbations/combined.csv")

# Used in table to show difference in effectiveness
baseline = df["trueB̃"][("baseline", 0)]
print(100 * (df[["trueB̃", "trueB̃_og"]] - baseline) / baseline)

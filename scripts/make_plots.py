import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import re
from os import path
from scipy import stats


# Get Args and  set global constants
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("indir")
    parser.add_argument("outdir")
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--no-hist', action='store_true')
    parser.add_argument('--no-rand', action='store_true')
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    TARGET_DIR = args.indir
    SAVE_DIR = args.outdir
    SHOW_PLOTS = args.show
    NO_HIST = args.no_hist
    NO_RAND = args.no_rand
    REVERSE = args.reverse

except Exception as e:
    print(e)
    TARGET_DIR = "results/perturbations/C0-V15-W8-D75-R0.05-E300-B2"
    SAVE_DIR = "results/figures"
    SHOW_PLOTS = True
    NO_HIST = False
    NO_RAND = False
    REVERSE = False

print(TARGET_DIR)
CORPUS_NUM = int(re.compile("C[0-9]+-V").search(TARGET_DIR).group(0)[1:-2])
BIAS_NUM = int(re.compile("B[0-9]+$").search(TARGET_DIR).group(0)[1:])
CORPUS = ({0: "Wiki", 1: "NYT"})[CORPUS_NUM]
BIAS = ({1: "WEAT 1", 2: "WEAT 2"})[BIAS_NUM]
SIZES = ({0: [10, 30, 100, 300, 1000], 1: [100, 300, 1000, 3000, 10000]})[CORPUS_NUM]

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

# Helpers
def mm2in(mm):
    return mm/25.4


def rename(name):
    if name == "correct":
        return "decrease"
    elif name == "aggravate":
        return "increase"
    return name


# Plotting Functions

# Histrogram of Diff Bias
def make_histogram(filename="diff_bias.csv", target_col="ΔBIF_μ"):
    df = pd.read_csv(path.join(TARGET_DIR, filename))
    x = df[target_col].values
    textstr = "N: {}\nμ: {:.5f}\nσ: {:.5f}".format(len(x), np.mean(x), np.std(x))
    props = dict(boxstyle='round', facecolor="white", alpha=1)
    # place a text box in upper left in axes coords
    fig = plt.figure(dpi=100, figsize=(6, 4))
    plt.hist(x, bins=100, log=True);
    plt.yscale('log', nonposy='clip')
    plt.ylabel("Number of Documents")
    plt.xlabel("Differential Bias of Removal (%)")
    # plt.title(CORPUS + " - " + BIAS)
    xl, xr = plt.xlim()
    yd, yu = plt.ylim()
    plt.text(0.9 * xl, 0.5 * yu, textstr, fontsize=16, verticalalignment='top',
             horizontalalignment='left', bbox=props)
    plt.tight_layout()
    plt.savefig(path.join(SAVE_DIR, "histogram_{}_{}.pdf".format(CORPUS, BIAS_NUM)),
                bbox_inches="tight", pad_inches=0)
    SHOW_PLOTS and plt.show()


# Comparison of approximation and
def make_comparision_plot(preds, trues, random=False):
    preds_grouped = preds.groupby(["pert_type", "pert_size", "pert_run"])
    trues_grouped = trues.groupby(["pert_type", "pert_size", "pert_run"])

    fig = plt.figure(dpi=150, figsize=(6, 5))

    # Y positions, and spacing
    positions = [("aggravate", s) for s in SIZES[::-1]]
    positions += [("baseline", 0)]
    positions += [("correct", s) for s in SIZES]
    D = 3  # vertical space between scenarios
    d = 0.8  # vertial space between ground truth and approximation
    random_scenarios = []

    # Baseline embedding
    base_weat = np.unique(preds["B"])
    base_mean = np.mean(base_weat)
    not random and print("WEAT - μ:", base_mean, "σ:", np.std(base_weat))
    if base_mean < 0.25 or REVERSE:
        positions = positions[::-1]  # Handle Wiki - WEAT2

    pos = D * positions.index(("baseline", 0)) + d/2
    y = pos * np.ones(len(base_weat))
    if not random:
        plt.scatter(base_weat, y, c='k', marker='.')
        plt.scatter(base_mean, pos, c='k', marker="d")
        plt.axvline(base_mean, c='k', ls="dotted")

    # Predictions
    # print("Targeted - approximation")
    trues_groups = dict(list(trues_grouped))
    first_preds = True
    first_trues = True
    i = 0
    for (pert_type, pert_size, pert_run), preds_group in preds_grouped:
        if ((pert_type == "random") if random else (pert_type != "random")):
            name = "-".join([str(x) for x in [pert_type, pert_size, pert_run]])
            random_scenarios.append(name)
            pos = D * i if random else D * positions.index((pert_type, pert_size))
            # Approximation
            preds_x = np.array(preds_group["IFB̃"])
            preds_u = np.mean(preds_x)
            preds_y = pos * np.ones(len(preds_x))
            plt.scatter(preds_x, preds_y, c='b', marker='.',
                        label=(first_preds and "approximation" or None))
            plt.scatter(preds_u, pos, c='b', marker="d",
                        label=(first_preds and "approx. mean" or None))
            first_preds = False
            i += 1
            # ground truth
            trues_group = trues_groups.get((pert_type, pert_size, pert_run))
            if trues_group is None:
                continue
            trues_x = np.array(trues_group["trueB̃"])
            trues_u = np.mean(trues_x)
            trues_y = (pos + d) * np.ones(len(trues_x))
            plt.scatter(trues_x, trues_y, c='r', marker='.',
                        label=(first_trues and "ground truth" or None))
            plt.scatter(trues_u, (pos + d), c='r', marker="d",
                        label=(first_trues and "gnd. truth mean" or None))
            first_trues = False
            # Statistical testing
            pAV = stats.ttest_ind(trues_x, preds_x, equal_var=False)[1]
            pBV = stats.ttest_ind(trues_x, base_weat, equal_var=False)[1]
            pAB = stats.ttest_rel(preds_x, base_weat)[1]
            print(pert_type, pert_size, "reject:", pBV < 0.05, "pBV:", pBV, "lower:", pBV < pAV, "pAV:", pAV)

    # Title etc.
    scenarios = [rename(name) + "-" + str(size) for (name, size) in positions]
    if base_mean >= 0.25:
        scenarios = scenarios[::-1]  # Handle Wiki - WEAT2
    plt.xlabel("WEAT effect size")
    if random:
        plt.yticks(D * np.arange(len(random_scenarios)), random_scenarios)
    else:
        plt.yticks(D * np.arange(len(positions)) + d/2, scenarios)
    # plt.ylabel("Scenario")
    plt.legend()
    # plt.title(CORPUS + " - " + BIAS)
    # Space ticks reasonably
    XLIM = plt.xlim((-2, 2))
    S = 2
    ax = plt.gca()
    ax.set_xticks(np.arange(np.ceil(S*XLIM[0])/S, (np.floor(S*XLIM[1]) + 1)/S, 1/S))
    plt.tight_layout()
    fig_name = "random_{}_{}.pdf" if random else "targeted_{}_{}.pdf"
    plt.savefig(path.join(SAVE_DIR, fig_name.format(CORPUS, BIAS_NUM)),
                bbox_inches="tight", pad_inches=0)
    SHOW_PLOTS and plt.show()



# Plot correlation of the means
def make_correlation_plot(preds, trues):
    preds_grouped = preds.groupby(["pert_type", "pert_size"])
    trues_grouped = trues.groupby(["pert_type", "pert_size"])
    fig = plt.figure(dpi=150, figsize=(6, 5))

    base_weat = np.unique(preds["B"])
    base_mean = np.mean(base_weat)
    plt.axvline(base_mean, c='k', ls="dotted")

    y_samples = []
    x_samples = []
    for (pert_type, pert_size), pred_group in preds_grouped:
        if (pert_type != "random"):
            y_samples.append(pred_group["IFB̃"])
            x_samples.append(trues[(trues["pert_type"] == pert_type) & (trues["pert_size"] == pert_size)]["trueB̃"])

    x_means = np.array([np.mean(x) for x in x_samples])
    y_means = np.array([np.mean(y) for y in y_samples])
    x_stds = np.array([np.std(x, ddof=1) for x in x_samples])
    y_stds = np.array([np.std(y, ddof=1) for y in y_samples])

    plt.errorbar(x_means, y_means, xerr=x_stds, yerr=y_stds, fmt="o", zorder=1000)

    # Correlation
    cor = np.corrcoef(y_means, x_means)[1, 0]
    print("r^2:", cor**2)
    A = np.vstack([x_means, np.ones(len(x_means))]).T
    a, b = np.linalg.lstsq(A, y_means)[0]
    print("slope:", a, "intercept:", b)
    x_means.sort()
    plt.plot(x_means, a*x_means + b, c='r', ls="dashed")

    # Title, format etc
    # plt.title(CORPUS + " - " + BIAS)
    plt.ylabel('Approximated Effect Size')
    plt.xlabel('Ground Truth Effect Size')
    ax = plt.gca()
    ax.set_aspect('equal')
    XMIN, XMAX = plt.xlim((-2, 2))
    YMIN, YMAX = plt.ylim((-2, 2))
    S = 2 # tick scale
    ax.set_xticks(np.arange(np.ceil(S*XMIN)/S, (np.floor(S*XMAX) + 1)/S, 1/S))
    ax.set_yticks(np.arange(np.ceil(S*YMIN)/S, (np.floor(S*YMAX) + 1)/S, 1/S))
    plt.tight_layout()
    plt.savefig(path.join(SAVE_DIR, "means_{}_{}.pdf".format(CORPUS, BIAS_NUM)),
                bbox_inches="tight", pad_inches=0)
    SHOW_PLOTS and plt.show()


# Main
if __name__ == "__main__":
    preds = pd.read_csv(path.join(TARGET_DIR, "predicted_change.csv"))
    trues = pd.read_csv(path.join(TARGET_DIR, "true_change.csv"))
    preds_grouped = preds.groupby(["pert_type", "pert_size"])
    trues_grouped = trues.groupby(["pert_type", "pert_size"])

    groups = dict(list(preds_grouped))
    groups.get(("correct", 10))

    not NO_HIST and make_histogram()
    make_comparision_plot(preds, trues, random=False)
    not NO_RAND and make_comparision_plot(preds, trues, random=True)
    make_correlation_plot(preds, trues)

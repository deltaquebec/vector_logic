"""figures from results csvs; matplotlib with pgf-friendly output

usage:
\tpython make_figures.py --train results_train --diag results_mcrae_cal results_wn_cal --out figures
writes figures/switch.pdf (fraction within tolerance vs d), figures/frontier.pdf (truth defect vs
distributional error), figures/defect_auc.pdf (per predicate defect vs probe auc).
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

LABEL = {"mcrae_glove": ("McRae, GloVe", "k", "-", "o"),
         "mcrae_w2v": ("McRae, word2vec", "k", "--", "s"),
         "wordnet_glove": ("WordNet, GloVe", "0.5", "-", "^"),
         "wordnet_w2v": ("WordNet, word2vec", "0.5", "--", "D")}


def load_train(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*_train.csv"))):
        name = os.path.basename(f).replace("_train.csv", "")
        out[name] = pd.read_csv(f)
    return out


def fig_switch(train, ranks, out, lam=1.0, tol_col="frac_exact"):
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    for name, df in train.items():
        lab, col, ls, mk = LABEL.get(name, (name, "k", "-", "o"))
        sub = df[np.isclose(df.lam, lam)].sort_values("d")
        ax.plot(sub.d, sub[tol_col], color=col, ls=ls, marker=mk, ms=3.5, lw=1.1, label=lab)
    seen = {}
    for name, r in ranks.items():
        _, col, _, _ = LABEL.get(name, (name, "k", "-", "o"))
        ax.axvline(r, color=col, ls=":", lw=0.8)
        # stack labels of ranks within 3 of one another
        key = round(r / 3)
        y = 0.55 - 0.07 * seen.get(key, 0)
        seen[key] = seen.get(key, 0) + 1
        ax.text(r + 2, y, str(r), fontsize=7, color=col, va="center")
    ax.set_xlabel("$d$")
    ax.set_ylabel("fraction within tolerance 0.05")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, max(df.d.max() for df in train.values()) + 10)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "switch.pdf"))
    plt.close(fig)


def fig_frontier(train, out, d=300):
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for name, df in train.items():
        lab, col, ls, mk = LABEL.get(name, (name, "k", "-", "o"))
        sub = df[(df.d == d) & (df.lam > 0) & (df.lam < 1)].sort_values("lam")
        ax.plot(sub.dist_error, sub.truth_defect, color=col, ls=ls, marker=mk, ms=3.5, lw=1.1, label=lab)
        for _, r in sub.iterrows():
            ax.annotate(f"{r.lam:g}", (r.dist_error, r.truth_defect), textcoords="offset points",
                        xytext=(3, 3), fontsize=6, color=col)
    ax.set_xlabel("distributional reconstruction error (relative)")
    ax.set_ylabel("joint truth defect (relative)")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "frontier.pdf"))
    plt.close(fig)


def fig_defect_auc(diag_dirs, out):
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for d in diag_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*_defects.csv"))):
            name = os.path.basename(f).replace("_defects.csv", "")
            lab, col, ls, mk = LABEL.get(name, (name, "k", "-", "o"))
            df = pd.read_csv(f)
            ycol = "defect_isotonic" if "defect_isotonic" in df else "defect_holdout"
            ax.scatter(df.probe_auc, df[ycol], s=9, color=col, marker=mk, alpha=0.7,
                       facecolors="none" if ls == "--" else col, linewidths=0.6, label=lab)
    ax.set_xlabel("probe AUC (cross validated)")
    ax.set_ylabel("held-out error, isotonic readout")
    ax.set_xlim(0.5, 1.02); ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="0.7", lw=0.6, ls=":")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "defect_auc.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="results_train")
    ap.add_argument("--diag", nargs="*", default=[])
    ap.add_argument("--ranks", default="mcrae_glove:76,mcrae_w2v:75,wordnet_glove:165,wordnet_w2v:138",
                    help="affine exact dimensions rank[T;1]-1 per cell, from the run_train log")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ranks = {k: int(v) for k, v in (x.split(":") for x in a.ranks.split(","))}
    train = load_train(a.train)
    if train:
        fig_switch(train, {k: v for k, v in ranks.items() if k in train}, a.out)
        fig_frontier(train, a.out)
    if a.diag:
        fig_defect_auc(a.diag, a.out)
    print("wrote", os.listdir(a.out))


if __name__ == "__main__":
    main()

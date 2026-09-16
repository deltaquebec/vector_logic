"""train geometries under mixed distributional and truth objective

usage:
\tpython run_train.py --glove G.txt --mcrae CONCS_FEATS_concstats_brm.txt --min_true 15 --out results_train
\tpython run_train.py --glove G.txt --wordnet 20000 --min_true 30 --out results_train
\tpython run_train.py --synthetic
"""
import argparse, os, sys, warnings, functools
warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)
import numpy as np
import pandas as pd
from vlcompress import (load_glove, load_word2vec, geometry_matrix, load_mcrae, truth_matrix,
                        augmented_rank, price_of_exactness, truth_floor, train_sweep, als_train,
                        per_predicate_defect)


def report(name, E, T, feats, out, ds, lams, iters):
    D, V = E.shape
    r = augmented_rank(T)
    print(f"\n{name}: D = {D}, V = {V}, |P| = {T.shape[0]}, rank[T;1] = {r}; affine exact dimension d* = {r - 1}")
    print("\tprice of exactness (fraction of pretrained variance retained), by dimension:")
    print("\t\td      unconstrained  exact   random-constraint")
    for d in sorted(set([x for x in ds if x >= r - 1] + [r - 1, D])):
        p = price_of_exactness(E, T, d)
        print(f"\t\t{d:4d}   {p['unconstrained']:.3f}          {p['exact']:.3f}   {p['random']:.3f}")
    print("\tclosed form truth floor (lam = 1) by dimension:")
    print("\t\t" + "  ".join(f"d={d}:{truth_floor(T, d):.3f}" for d in ds))
    rows = train_sweep(E, T, ds, lams, iters=iters)
    df = pd.DataFrame(rows, columns=["d", "lam", "truth_defect", "dist_error", "truth_floor", "frac_exact"])
    df.to_csv(os.path.join(out, f"{name}_train.csv"), index=False)
    print("\talternating least squares, final truth defect / distributional error / fraction of predicates exact (< 0.05):")
    hdr = "\t\t d   " + "".join(f"lam={l:<5g}          " for l in lams)
    print(hdr)
    for d in ds:
        sub = df[df.d == d]
        cells = "".join(f"{t:.3f}/{e:.3f}/{f:.2f}   " for t, e, f in zip(sub.truth_defect, sub.dist_error, sub.frac_exact))
        print(f"\t\t{d:4d}  {cells}")
    # detail at d = D, lam = 0.5: which predicates remain inexact
    H, hist = als_train(E, T, D, 0.5, iters=iters)
    pdf = per_predicate_defect(H, T)
    order = np.argsort(-pdf)
    print(f"\tat d = {D}, lam = 0.5: median predicate defect {np.nanmedian(pdf):.3f}; worst:")
    for i in order[:5]:
        print(f"\t\t{feats[i]:30s} {pdf[i]:.3f}")


def synthetic(V=800, D=100, m=20, seed=0):
    rng = np.random.default_rng(seed)
    T = (rng.random((m, V)) < 0.3).astype(float)
    E = rng.standard_normal((D, V)) + 0.5 * rng.standard_normal((D, 8)) @ rng.standard_normal((8, V))
    return E, T, [f"f{k}" for k in range(m)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove"); ap.add_argument("--w2v")
    ap.add_argument("--mcrae"); ap.add_argument("--wordnet", type=int, default=0)
    ap.add_argument("--wn_min_members", type=int, default=30)
    ap.add_argument("--min_prod_freq", type=int, default=5)
    ap.add_argument("--min_true", type=int, default=15)
    ap.add_argument("--ds", default="10,20,40,80,120,160,200,300")
    ap.add_argument("--lams", default="0,0.1,0.5,0.9,1")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out", default="results_train")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ds = [int(x) for x in a.ds.split(",")]
    lams = [float(x) for x in a.lams.split(",")]
    if a.synthetic:
        E, T, feats = synthetic()
        report("synthetic", E, T, feats, a.out, [5, 10, 20, 21, 30, 60, 100], lams, a.iters)
        return
    embs = []
    if a.glove:
        embs.append(("glove", load_glove(a.glove, vocab=None)))
    if a.w2v:
        embs.append(("w2v", load_word2vec(a.w2v, vocab=None)))
    if not embs:
        sys.exit("need --glove or --w2v")
    for ename, vecs in embs:
        if a.mcrae:
            df = load_mcrae(a.mcrae, a.min_prod_freq)
            ents = sorted(set(df["word"]))
            E, ents, _ = geometry_matrix(vecs, ents)
            T, feats = truth_matrix(df, ents, min_pos=a.min_true, min_neg=a.min_true)
            report(f"mcrae_{ename}", E, T, feats, a.out, [d for d in ds if d <= E.shape[0]], lams, a.iters)
        if a.wordnet:
            from vlcompress.norms import load_wordnet, frequent_nouns
            nouns = frequent_nouns(vecs, n=a.wordnet)
            df = load_wordnet(nouns, min_members=a.wn_min_members)
            ents = sorted(set(df["word"]))
            E, ents, _ = geometry_matrix(vecs, ents)
            T, feats = truth_matrix(df, ents, min_pos=a.min_true, min_neg=a.min_true)
            report(f"wordnet_{ename}", E, T, feats, a.out, [d for d in ds if d <= E.shape[0]], lams, a.iters)


if __name__ == "__main__":
    main()
